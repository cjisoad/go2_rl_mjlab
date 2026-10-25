import copy
import math
from types import SimpleNamespace
import torch
from src.tasks.go2.amp.features import amp_features, terminal_next_observation
from src.tasks.go2.amp.discriminator import Discriminator
from src.tasks.go2.amp.algorithm import mix_rewards
from src.tasks.go2.amp.replay import ReplayBuffer


def test_feature_order():
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    data = SimpleNamespace(
        joint_pos=torch.arange(12.0).reshape(1, 12),
        joint_vel=torch.arange(12.0, 24.0).reshape(1, 12),
        root_link_lin_vel_b=torch.tensor([[24.0, 25.0, 26.0]]),
        root_link_ang_vel_b=torch.tensor([[27.0, 28.0, 29.0]]),
    )
    asset = SimpleNamespace(data=data, joint_names=names)
    env = SimpleNamespace(scene={"robot": asset})
    result = amp_features(env)
    assert result[0, :12].tolist() == [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    assert result[0, 24:].tolist() == [24, 25, 26, 27, 28, 29]


def test_reward_and_r1_numeric():
    d = Discriminator(4, [2], 1.0, use_minibatch_std=False)
    for p in d.parameters():
        p.data.zero_()
    x = torch.ones(3, 2)
    assert torch.allclose(d.predict_reward(x, x), torch.full((3,), math.log(2)))
    logits = d(torch.cat((x, x), -1))
    loss, penalty = d.compute_loss(logits, logits, (x, x), (x, x))
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert penalty.item() == 0
    for param in d.parameters():
        param.data.fill_(1.0)
    scores = d(torch.cat((x, x), -1))
    _, nonzero_penalty = d.compute_loss(scores, scores, (x, x), (x, x))
    # Two hidden units each contribute 1 to each of four input derivatives.
    assert abs(nonzero_penalty.item() - 80.0) < 1e-6
    assert torch.allclose(
        mix_rewards(torch.tensor([2.0]), torch.tensor([3.0])), torch.tensor([2.4])
    )


def test_terminal_replaces_reset_and_replay_wraps():
    nxt = torch.zeros(3, 2)
    terminal = torch.ones(3, 2) * 7
    actual = terminal_next_observation(
        nxt, torch.tensor([False, True, False]), {"amp_terminal": terminal}
    )
    assert actual.tolist() == [[0, 0], [7, 7], [0, 0]]
    buf = ReplayBuffer(2, 4)
    buf.insert(torch.arange(6.0).reshape(3, 2), torch.ones(3, 2))
    buf.insert(torch.ones(3, 2) * 9, torch.ones(3, 2) * 10)
    assert len(buf) == 4 and buf.step == 2
    samples = list(buf.feed_forward_generator(3, 4))
    assert len(samples) == 3


def test_native_ppo_rollout_and_checkpoint():
    from dataclasses import asdict
    from tensordict import TensorDict
    from src.tasks.go2.amp.config import amp_runner_cfg
    from src.tasks.go2.amp.algorithm import AMPPPO

    obs = TensorDict(
        {
            "actor": torch.randn(4, 6),
            "critic": torch.randn(4, 8),
            "amp": torch.randn(4, 30),
        },
        batch_size=[4],
    )
    env = SimpleNamespace(
        num_envs=4, num_actions=2, unwrapped=SimpleNamespace(step_dt=0.02)
    )
    cfg = asdict(amp_runner_cfg())
    cfg["multi_gpu"] = None
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"]["num_learning_epochs"] = 1
    cfg["algorithm"]["num_mini_batches"] = 1
    cfg["actor"]["hidden_dims"] = [8]
    cfg["critic"]["hidden_dims"] = [8]
    for key in ("actor", "critic"):
        cfg[key].pop("cnn_cfg")
        if cfg[key]["distribution_cfg"] is None:
            cfg[key].pop("distribution_cfg")
    alg = AMPPPO.construct_algorithm(obs, env, cfg, "cpu")
    initial = copy.deepcopy(alg.discriminator.state_dict())
    for _ in range(2):
        alg.act(obs)
        alg.process_env_step(
            obs,
            torch.ones(4),
            torch.tensor([0, 1, 0, 0]),
            {"amp_terminal": torch.ones(4, 30) * 7},
        )
    assert torch.equal(alg.amp_storage.next_states[1], torch.ones(30) * 7)
    alg.compute_returns(obs)
    stats = alg.update()
    assert math.isfinite(stats["amp"])
    assert any(
        not torch.equal(initial[k], v)
        for k, v in alg.discriminator.state_dict().items()
    )
    state = copy.deepcopy(alg.save())
    alg.discriminator.linear.weight.data.zero_()
    alg.load(state, None, True)
    assert torch.equal(
        state["amp"]["discriminator"]["linear.weight"], alg.discriminator.linear.weight
    )
    assert alg.amp_storage.num_samples == 8
    assert torch.equal(alg.amp_storage.next_states, state["amp_replay"]["next_states"])
    # Export the native actor on CPU, as the runner does when saving.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        export = alg.actor.as_onnx(verbose=False)
        export.eval()
        path = Path(directory) / "policy.onnx"
        torch.onnx.export(
            export,
            export.get_dummy_inputs(),
            str(path),
            opset_version=18,
            input_names=export.input_names,
            output_names=export.output_names,
            dynamo=False,
        )
        assert path.stat().st_size > 0


def test_motion_weights_and_config():
    from src.tasks.go2.amp.config import amp_env_cfg, amp_runner_cfg
    from src.tasks.go2.amp.algorithm import load_demonstrations

    data = load_demonstrations("cpu", 0.02)
    assert data.all_obs.shape[1] == 30
    assert torch.allclose(data.dataset_weights, torch.tensor([1.0] * 6 + [0.5] * 4) / 8)
    assert torch.isfinite(data.all_obs).all()
    cfg = amp_env_cfg()
    assert "amp" in cfg.observations and "amp_terminal_capture" in cfg.rewards
    assert amp_runner_cfg().logger == "tensorboard"


def test_source_policy_and_actuator_contract():
    from src.tasks.go2.amp.config import amp_env_cfg

    cfg = amp_env_cfg()
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel",
        "projected_gravity",
        "command",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert list(cfg.observations["critic"].terms) == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "command",
        "joint_pos",
        "joint_vel",
        "joint_effort",
        "actions",
    ]
    assert cfg.observations["actor"].terms["base_ang_vel"].scale == 0.2
    assert cfg.observations["actor"].terms["joint_vel"].scale == 0.05
    assert cfg.scene.entities["robot"].articulation.actuators[0].stiffness == 25.0
    assert cfg.scene.entities["robot"].articulation.actuators[0].damping == 0.5


def test_source_tracking_ignores_vertical_and_roll_pitch_velocity():
    from src.tasks.go2.amp.features import track_xy, track_yaw

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[0.0, 0.0, 9.0]]),
            root_link_ang_vel_b=torch.tensor([[9.0, 9.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(1, 3)),
    )
    assert track_xy(env).item() == 1.0
    assert track_yaw(env).item() == 1.0


def test_source_command_curriculum():
    from src.tasks.go2.amp.features import command_curriculum

    ranges = SimpleNamespace(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
        ),
        reward_manager=SimpleNamespace(
            _episode_sums={"track_linear_velocity": torch.tensor([1600.0])},
            get_term_cfg=lambda name: SimpleNamespace(weight=75.0),
        ),
        max_episode_length_s=20.0,
        max_episode_length=1000,
        common_step_counter=1000,
        device="cpu",
    )
    command_curriculum(env, torch.tensor([0]))
    assert abs(ranges.lin_vel_x[1] - 0.2) < 1e-6
    assert abs(ranges.lin_vel_y[0] + 0.2) < 1e-6
    from src.tasks.go2.amp.config import amp_env_cfg

    cfg = amp_env_cfg()
    assert cfg.commands["twist"].resampling_time_range == (10.0, 10.0)
    assert cfg.commands["twist"].rel_standing_envs == 0.1
    assert cfg.events["reset_robot_joints"].params["velocity_range"] == (-1.0, 1.0)


def test_source_physics_events_and_termination_contract():
    from src.tasks.go2.amp.config import amp_env_cfg

    cfg = amp_env_cfg()
    assert cfg.sim.mujoco.timestep == 0.005
    assert cfg.decimation == 4
    assert cfg.scene.entities["robot"].articulation.actuators[0].frictionloss == 0.01
    mass = cfg.events["base_mass"]
    assert mass.mode == "startup" and mass.params["operation"] == "add"
    assert mass.params["ranges"] == (-1.0, 3.0)
    assert mass.params["asset_cfg"].body_names == ("base_link",)
    assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == (
        ".*_collision.*",
    )
    assert set(cfg.terminations) == {"time_out", "base_contact", "bad_orientation"}
    assert cfg.terminations["base_contact"].params["force_threshold"] == 1.0
    assert cfg.terminations["bad_orientation"].params["limit_angle"] == 0.8
    for name, lo, hi in [
        ("base_ang_vel", -0.2, 0.2),
        ("projected_gravity", -0.05, 0.05),
        ("joint_pos", -0.01, 0.01),
        ("joint_vel", -1.5, 1.5),
    ]:
        noise = cfg.observations["actor"].terms[name].noise
        assert noise.n_min == lo and noise.n_max == hi


def test_processed_position_action_clipping():
    from src.tasks.go2.amp.actions import AMPJointPositionAction

    action = object.__new__(AMPJointPositionAction)
    action._raw_actions = torch.zeros(1, 3)
    action._scale = 0.25
    action._offset = torch.tensor([[0.0, 1.0, -1.0]])
    raw = torch.tensor([[800.0, -800.0, 2.0]])
    action.process_actions(raw)
    assert action._processed_actions.tolist() == [[100.0, -100.0, -0.5]]
    assert torch.equal(action._raw_actions, raw)


def test_self_collision_and_restitution_adaptation():
    from src.tasks.go2.amp.config import amp_env_cfg
    from src.tasks.go2.amp.features import restitution_damping, randomize_restitution

    cfg = amp_env_cfg()
    for collision in cfg.scene.entities["robot"].collisions:
        assert collision.contype == 1 and collision.conaffinity == 1
    assert "restitution" in cfg.events
    values = restitution_damping(torch.tensor([0.0, math.exp(-math.pi)]))
    assert torch.allclose(values, torch.tensor([1.0, 2**-0.5]), atol=1e-6)
    robot = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([1, 3])))
    model = SimpleNamespace(geom_solref=torch.zeros(2, 4, 2))
    env = SimpleNamespace(
        scene={"robot": robot},
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(model=model),
    )
    asset = SimpleNamespace(name="robot", geom_ids=slice(None))
    randomize_restitution(env, None, asset)
    assert (model.geom_solref[:, [1, 3], 1] > 0).all()
    assert (model.geom_solref[:, [1, 3], 1] <= 1).all()
    assert model.geom_solref[:, [0, 2], 1].sum() == 0


def test_source_play_and_unfiltered_base_contacts():
    from src.tasks.go2.amp.config import amp_env_cfg

    cfg = amp_env_cfg(play=True)
    assert cfg.scene.num_envs == 32
    assert cfg.episode_length_s == 20.0
    assert cfg.observations["actor"].enable_corruption
    assert cfg.commands["twist"].ranges.lin_vel_x == (-1.0, 1.0)
    assert cfg.commands["twist"].ranges.lin_vel_y == (-0.4, 0.4)
    assert "push_robot" in cfg.events
    assert cfg.events["push_robot"].interval_range_s == (5.0, 10.0)
    assert "lin_vel_cmd_levels" in cfg.curriculum
    for source in (amp_env_cfg(), cfg):
        sensor = next(
            sensor
            for sensor in source.scene.sensors
            if sensor.name == "amp_base_contact"
        )
        assert sensor.secondary is None


def load_tests(loader, tests, pattern):
    import unittest

    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith("test_")
    )


if __name__ == "__main__":
    import unittest

    unittest.main()

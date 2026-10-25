import torch
from tensordict import TensorDict


def observations(n=4):
    return TensorDict(
        {
            "actor": torch.randn(n, 45),
            "obs_history": torch.randn(n, 225),
            "privileged": torch.randn(n, 202),
        },
        [n],
    )


def test_context_training_stochastic_inference_deterministic():
    from src.tasks.go2.dreamwaq.rl import DreamWaQActor

    obs = observations()
    actor = DreamWaQActor(
        obs,
        {"actor": ["actor"]},
        "actor",
        12,
        hidden_dims=[16],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    assert torch.equal(actor(obs), actor(obs))
    assert not torch.equal(
        actor.context(obs["obs_history"], True)[0],
        actor.context(obs["obs_history"], True)[0],
    )
    losses = actor.estimator_losses(obs, beta=1.0, num_mini_batches=4)
    assert set(losses) == {"velocity", "reconstruction", "kl"}
    sum(losses.values()).backward()
    for name in [
        "mean_velocity",
        "logvar_velocity",
        "mean_latent",
        "logvar_latent",
        "decoder",
    ]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in getattr(actor, name).parameters()
        )


def test_export_uses_history_and_matches_inference():
    from src.tasks.go2.dreamwaq.rl import DreamWaQActor

    obs = observations()
    actor = DreamWaQActor(obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=[16])
    export = actor.as_onnx(False)
    assert torch.allclose(export(obs["actor"], obs["obs_history"]), actor(obs))
    assert export.input_names == ["obs", "obs_history"]


def test_history_frame_scales_and_clips_before_scale():
    from src.tasks.go2.dreamwaq.mdp import scale_frame

    frame = torch.ones(1, 45) * 200
    out = scale_frame(frame, noise=False)
    assert torch.equal(out[0, :3], torch.ones(3) * 20)
    assert torch.equal(out[0, 21:33], torch.ones(12) * 5)
    assert torch.equal(out[0, 3:21], torch.ones(18) * 100)


def test_manager_history_is_frame_major_oldest_first_and_reset_backfills():
    from types import SimpleNamespace
    from mjlab.managers import (
        ObservationManager,
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    env = SimpleNamespace(num_envs=2, device="cpu", frame=torch.ones(2, 45))

    def frame(env):
        return env.frame.clone()

    manager = ObservationManager(
        {
            "obs_history": ObservationGroupCfg(
                terms={"frame": ObservationTermCfg(func=frame)},
                history_length=5,
                flatten_history_dim=True,
                concatenate_terms=True,
            )
        },
        env,
    )
    manager.compute(update_history=True)
    for step in range(2, 7):
        env.frame.fill_(step)
        history = manager.compute(update_history=True)["obs_history"].reshape(2, 5, 45)
    assert torch.equal(history[0, :, 0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    manager.reset(torch.tensor([0]))
    env.frame.fill_(7)
    history = manager.compute(update_history=True)["obs_history"].reshape(2, 5, 45)
    assert torch.equal(history[0, :, 0], torch.ones(5) * 7)
    assert torch.equal(history[1, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0]))


def test_native_ppo_update_checkpoint_includes_estimator_and_optimizer():
    from src.tasks.go2.dreamwaq.rl import DreamWaQActor, DreamWaQPPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    obs = observations()
    groups = {"actor": ["actor"], "critic": ["actor", "privileged"]}

    def build():
        actor = DreamWaQActor(
            obs,
            groups,
            "actor",
            12,
            hidden_dims=[16],
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
        critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16])
        return DreamWaQPPO(
            actor,
            critic,
            RolloutStorage("rl", 4, 2, obs, [12], "cpu"),
            num_learning_epochs=1,
            num_mini_batches=1,
            device="cpu",
        )

    alg = build()
    before = alg.actor.mean_velocity.weight.detach().clone()
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(4), torch.zeros(4), {})
        alg.compute_returns(obs)
    loss = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in loss.values())
    assert not torch.equal(before, alg.actor.mean_velocity.weight)
    assert set(loss) >= {
        "estimator_velocity",
        "estimator_reconstruction",
        "estimator_kl",
    }
    restored = build()
    restored.load(alg.save(), None, True)
    assert torch.equal(restored.actor(obs), alg.actor(obs))
    assert len(restored.optimizer.state) == len(alg.optimizer.state) > 0


def test_onnx_export_embeds_estimator(tmp_path=None):
    import tempfile
    from pathlib import Path
    import onnx
    from src.tasks.go2.dreamwaq.rl import DreamWaQActor

    obs = observations()
    actor = DreamWaQActor(obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=[16])
    export = actor.as_onnx(False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.onnx"
        torch.onnx.export(
            export,
            export.get_dummy_inputs(),
            str(path),
            input_names=export.input_names,
            output_names=export.output_names,
            opset_version=18,
            dynamo=False,
        )
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        assert [x.name for x in model.graph.input] == ["obs", "obs_history"]
        assert any("mean_velocity" in p.name for p in model.graph.initializer)


def test_native_observation_fields_and_privileged_velocity_layout():
    from types import SimpleNamespace
    from src.tasks.go2.dreamwaq import mdp

    data = SimpleNamespace(
        root_link_ang_vel_b=torch.ones(2, 3) * 2,
        projected_gravity_b=torch.ones(2, 3) * 3,
        joint_pos=torch.ones(2, 12) * 5,
        default_joint_pos=torch.ones(2, 12),
        joint_vel=torch.ones(2, 12) * 6,
        actuator_force=torch.ones(2, 12) * 7,
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=data,
                actuators=[
                    SimpleNamespace(
                        target_ids=torch.arange(12), ctrl_ids=torch.arange(12)
                    )
                ],
            )
        },
        command_manager=SimpleNamespace(get_command=lambda name: torch.ones(2, 3) * 4),
        action_manager=SimpleNamespace(action=torch.ones(2, 12) * 8),
    )
    frame = mdp.proprioception(env, noise=False)
    assert frame.shape == (2, 45)
    assert torch.allclose(frame[:, :3], torch.ones(2, 3) * 0.4)
    assert torch.equal(frame[:, 9:21], torch.ones(2, 12) * 4)
    assert torch.allclose(frame[:, 21:33], torch.ones(2, 12) * 0.3)
    assert torch.equal(mdp.effort(env), data.actuator_force)


def test_baseline_factory_preserves_dimensions_rewards_and_defaults():
    from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg
    from src.tasks.go2.dreamwaq.rl_cfg import dreamwaq_runner_cfg

    train, play = dreamwaq_env_cfg(), dreamwaq_env_cfg(True)
    assert train.scene.num_envs == 4096 and play.scene.num_envs == 32
    assert train.observations["obs_history"].history_length == 5
    assert list(train.observations["privileged"].terms) == [
        "base_lin_vel",
        "joint_effort",
        "height_scan",
    ]
    assert len(train.rewards) == 16
    assert train.rewards["track_linear"].weight == 1.5
    assert train.rewards["air_variance"].weight == -1
    assert train.events["push_robot"].interval_range_s == (5.0, 10.0)
    assert train.scene.terrain.terrain_generator.num_rows == 5
    assert play.scene.terrain.terrain_generator.num_cols == 7
    rl = dreamwaq_runner_cfg()
    assert rl.obs_groups["critic"] == ("actor", "privileged")
    assert rl.logger == "tensorboard" and rl.experiment_name == "go2_dreamwaq"
    assert rl.algorithm.beta == 1.0 and rl.max_iterations == 5000


def test_source_processed_joint_target_clipping():
    from src.tasks.go2.dreamwaq.actions import DreamWaQJointPositionAction

    action = object.__new__(DreamWaQJointPositionAction)
    action._raw_actions = torch.zeros(1, 12)
    action._scale = 0.25
    action._offset = torch.ones(1, 12)
    action.process_actions(torch.ones(1, 12) * 1000)
    assert torch.equal(action._processed_actions, torch.ones(1, 12) * 100)
    assert torch.equal(action._raw_actions, torch.ones(1, 12) * 1000)


def test_undesired_contacts_ignore_feet_and_sum_geoms_before_threshold():
    from src.tasks.go2.dreamwaq.mdp import undesired_contact_count

    names = [
        "FL_calf1_collision",
        "FL_calf2_collision",
        "FL_foot_collision",
        "FR_thigh_collision",
    ]
    forces = torch.zeros(2, 4, 3, 3)
    forces[0, 2, :, 2] = 100  # Foot support alone must incur no calf penalty.
    forces[1, 0, :, 0] = 0.7
    forces[1, 1, :, 0] = (
        0.7  # One calf body total exceeds threshold, neither geom does.
    )
    assert torch.equal(undesired_contact_count(forces, names), torch.tensor([0, 1]))
    forces[1, 1, :, 0] = -0.7  # Opposing geometry forces cancel before the norm.
    assert torch.equal(undesired_contact_count(forces, names), torch.tensor([0, 0]))
    forces[0, 3, 0, 1] = 2  # History still catches transient thigh contact.
    assert torch.equal(undesired_contact_count(forces, names), torch.tensor([1, 0]))


def test_source_self_collisions_and_all_contact_reporting_enabled():
    from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg

    cfg = dreamwaq_env_cfg()
    assert all(
        c.contype == 1 and c.conaffinity == 1
        for c in cfg.scene.entities["robot"].collisions
    )
    for sensor in cfg.scene.sensors:
        if sensor.name in ("feet_ground_contact", "base_contact", "undesired_contact"):
            assert sensor.secondary is None
    undesired = next(s for s in cfg.scene.sensors if s.name == "undesired_contact")
    assert undesired.primary.mode == "geom"
    assert ".*_foot_collision" in undesired.primary.exclude


def test_effort_uses_native_actuator_force_in_joint_order():
    from types import SimpleNamespace
    from mjlab.entity.data import EntityData
    from src.tasks.go2.dreamwaq.mdp import effort

    data = object.__new__(EntityData)
    data.indexing = SimpleNamespace(ctrl_ids=torch.tensor([0, 1, 2]))
    data.data = SimpleNamespace(actuator_force=torch.tensor([[10.0, 20.0, 30.0]]))
    data.default_joint_pos = torch.zeros(1, 3)
    # Real native EntityData.joint_torques is unimplemented; deliberately retain it.
    robot = SimpleNamespace(
        data=data,
        actuators=[
            SimpleNamespace(
                target_ids=torch.tensor([2, 0, 1]), ctrl_ids=torch.tensor([0, 1, 2])
            )
        ],
    )
    env = SimpleNamespace(scene={"robot": robot})
    assert torch.equal(effort(env), torch.tensor([[20.0, 30.0, 10.0]]))


def test_native_command_constructor_and_source_play_semantics():
    from types import SimpleNamespace
    from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg

    for play in (False, True):
        cfg = dreamwaq_env_cfg(play)
        env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": object()})
        command = cfg.commands["twist"].build(env)
        assert command.command.shape == (2, 3)
        assert cfg.commands["twist"].ranges.heading is None
        assert cfg.episode_length_s == 20.0
        assert cfg.observations["actor"].terms["proprioception"].params["noise"]
        assert "push_robot" in cfg.events
        if play:
            assert cfg.commands["twist"].ranges.lin_vel_x == (-1.0, 1.0)
            assert cfg.commands["twist"].ranges.lin_vel_y == (-0.4, 0.4)


def test_global_gradient_norm_clips_actor_estimator_and_critic_together():
    from src.tasks.go2.dreamwaq.rl import DreamWaQPPO

    algorithm = object.__new__(DreamWaQPPO)
    algorithm.actor = torch.nn.Linear(1, 1, bias=False)
    algorithm.critic = torch.nn.Linear(1, 1, bias=False)
    algorithm.actor.weight.grad = torch.tensor([[3.0]])
    algorithm.critic.weight.grad = torch.tensor([[4.0]])
    algorithm.max_grad_norm = 1.0
    algorithm.clip_gradients()
    assert torch.allclose(algorithm.actor.weight.grad, torch.tensor([[0.6]]), atol=1e-6)
    assert torch.allclose(
        algorithm.critic.weight.grad, torch.tensor([[0.8]]), atol=1e-6
    )


def test_restitution_randomization_native_mapping_and_event():
    from src.tasks.go2.dreamwaq.mdp import restitution_damping
    from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg

    e = torch.tensor([0.0, 0.15])
    ratio = restitution_damping(e)
    assert ratio[0] == 1.0
    assert torch.allclose(
        torch.exp(-torch.pi * ratio[1] / torch.sqrt(1 - ratio[1] ** 2)), e[1], atol=1e-6
    )
    cfg = dreamwaq_env_cfg()
    assert cfg.events["restitution"].mode == "startup"


def test_training_terrain_sap_workload_fits_backend_int32_scan():
    import mujoco
    import numpy as np
    from mjlab.scene import Scene
    from src.tasks.go2.dreamwaq.env_cfg import dreamwaq_env_cfg

    cfg = dreamwaq_env_cfg()
    num_worlds = cfg.scene.num_envs
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    data.qpos[:3] = scene.env_origins[0].numpy() + np.array([0.0, 0.0, 0.4])
    mujoco.mj_forward(model, data)
    # Match the installed MuJoCo-Warp SAP projection and pre-filter scan.
    # Even static/static and visual pairs enter this scan before filtering.
    direction = np.array([0.5935, 0.7790, 0.1235])
    direction /= np.linalg.norm(direction)
    radii = (
        np.where(model.geom_rbound == 0, 1e10, model.geom_rbound) + model.geom_margin
    )
    center = data.geom_xpos @ direction
    lower, upper = center - radii, center + radii
    static = model.body_weldid[model.geom_bodyid] == 0
    static_lower, static_upper = lower[static], upper[static]
    order = np.argsort(static_lower)
    static_pairs = int(
        (
            np.searchsorted(static_lower[order], static_upper[order], side="right")
            - np.arange(len(order))
            - 1
        ).sum()
    )
    dynamic_count = int((~static).sum())
    cross_bound = 0
    for radius in radii[~static]:
        # Maximum overlap of expanded static intervals bounds this dynamic
        # geometry's candidates at ANY position, independent of joint pose.
        events = np.concatenate((static_lower - radius, static_upper + radius))
        signs = np.concatenate(
            (
                np.ones(len(static_lower), dtype=int),
                -np.ones(len(static_lower), dtype=int),
            )
        )
        event_order = np.lexsort((-signs, events))  # Inclusive intervals: starts first.
        cross_bound += int(np.cumsum(signs[event_order]).max())
    pair_bound = static_pairs + cross_bound + dynamic_count * (dynamic_count - 1) // 2
    # _sap_range includes one extra sorted geometry unless at the last entry.
    scan_bound = (pair_bound + model.ngeom - 1) * num_worlds
    # The int32 work cursor increments by nsweep after its final work item.
    cursor_bound = scan_bound + 5 * num_worlds * model.ngeom - 1
    assert cursor_bound <= np.iinfo(np.int32).max, (
        f"SAP int32 work cursor overflows: bound={cursor_bound} for {num_worlds} worlds"
    )


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

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1] / "src/tasks/go2/standup"


def load(name):
    path = ROOT / (name + ".py")
    assert path.exists(), f"Missing native standup module: {path}"
    spec = importlib.util.spec_from_file_location("standup_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_tracking_uses_upright_axes_and_batch_height_gate():
    reward = load("source_rewards").SourceRewards()
    reward.cfg = SimpleNamespace(
        rewards=SimpleNamespace(tracking_sigma=0.25, base_height_target=0.52)
    )
    reward.commands = torch.tensor([[0.3, 0.2, 0.4], [0.3, 0.2, 0.4]])
    reward.base_lin_vel = torch.tensor([[0.0, 0.2, -0.3], [0.0, 0.2, 0.3]])
    reward.base_ang_vel = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.4]])
    reward.rew_hanstand = torch.tensor(0.7)
    assert torch.equal(reward._reward_tracking_lin_vel(), torch.zeros(2))
    reward.root_states = torch.zeros(2, 13)
    reward.root_states[:, 2] = 0.52
    reward.measured_heights = 0.0
    assert torch.equal(reward._reward_base_height(), torch.ones(2))
    torch.testing.assert_close(
        reward._reward_tracking_lin_vel(),
        torch.tensor([1.0, torch.exp(torch.tensor(-1.44))]),
    )
    torch.testing.assert_close(
        reward._reward_tracking_ang_vel(),
        torch.tensor([1.0, torch.exp(torch.tensor(-0.64))]),
    )


def test_front_feet_reward_and_rear_contact_semantics():
    reward = load("source_rewards").SourceRewards()
    reward.feet_name_reward_indices = [0, 1]
    reward.contact_foot_indices = [2, 3]
    reward.contact_forces = torch.zeros(2, 4, 3)
    reward.contact_forces[:, 2, 2] = 2.0
    reward.contact_forces[1, 0, 2] = 2.0
    reward.rew_hanstand = torch.tensor(1.0)
    torch.testing.assert_close(
        reward._reward_handstand_feet_on_air(), torch.tensor([1.0, 0.0])
    )
    assert reward._reward_contact().all()


def test_reset_pose_command_and_reward_config_are_source_specific():
    from src.tasks.go2.standup.env_cfg import standup_env_cfg

    cfg = standup_env_cfg()
    assert cfg.scene.entities["robot"].init_state.pos == (0.0, 0.0, 0.42)
    assert cfg.commands["base_velocity"].ranges.lin_vel_x == (-0.2, 0.6)
    assert cfg.decimation == 4 and cfg.sim.mujoco.timestep == 0.005
    assert cfg.rewards["handstand_feet_height_exp"].weight == 5.0
    assert list(cfg.rewards) == sorted(cfg.rewards)
    assert cfg.scene.entities["robot"].articulation.actuators[0].stiffness == 40.0


def test_command_distribution_and_heading_feedback():
    from src.tasks.go2.standup.mdp import SourceVelocityCommandCfg

    n = 20000
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4)
        )
    )
    env = SimpleNamespace(device="cpu", num_envs=n, scene={"robot": robot})
    command = SourceVelocityCommandCfg().build(env)
    torch.manual_seed(7)
    command._resample_command(torch.arange(n))
    values = command.source_commands
    assert values[:, 0].min() >= -0.2 and values[:, 0].max() <= 0.6
    assert (values[:, 1] == 0).all()
    nonzero = values[:, 0] != 0
    assert (values[nonzero, 0].abs() > 0.1).all()
    assert 0.18 < (values == 0).all(dim=1).float().mean() < 0.22
    command.source_commands[:, 3] = 3.0
    command._update_command()
    assert (command.command[:, 2] == 1.0).all()


def test_source_reset_randomizes_joints_and_root_velocity():
    from src.tasks.go2.standup.mdp import reset_source, JOINT_NAMES
    from src.tasks.go2.standup.source_config import Go2_Handstand_Cfg_Yu as Cfg

    n = 128
    defaults = torch.tensor(
        [Cfg.init_state.default_joint_angles[k] for k in JOINT_NAMES]
    ).expand(n, 12)

    class Robot:
        is_fixed_base = False
        data = SimpleNamespace(
            default_joint_pos=defaults,
            joint_pos=defaults.clone(),
            default_root_state=torch.zeros(n, 13),
        )

        def write_root_link_pose_to_sim(self, pose, env_ids):
            self.root = pose.new_zeros(n, 13)
            self.root[:, :7] = pose

        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            self.root[:, 7:] = velocity

        def write_joint_state_to_sim(self, pos, vel, env_ids):
            self.pos, self.vel = pos, vel

    robot = Robot()
    robot.data.default_root_state[:, 2] = 0.42
    robot.data.default_root_state[:, 3] = 1.0

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(device="cpu", num_envs=n, scene=scene)
    reset_source(env, torch.arange(n))
    torch.testing.assert_close(robot.root[:, :7], robot.data.default_root_state[:, :7])
    assert robot.root[:, 7:].abs().max() <= 0.5
    ratio = robot.pos / defaults
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5
    assert not robot.vel.any()


def test_delayed_pd_targets_retain_actions_across_reset():
    from src.tasks.go2.standup.actions import SourceDelayedJointPositionActionCfg
    from src.tasks.go2.standup.mdp import JOINT_NAMES

    class Robot:
        data = SimpleNamespace(default_joint_pos=torch.zeros(2, 12))

        def find_joints(self, names, preserve_order):
            return list(range(12)), list(names)

        def set_joint_position_target(self, target, joint_ids):
            self.target = target

    robot = Robot()
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        scene={"robot": robot},
        cfg=SimpleNamespace(decimation=4),
        _standup_v2_motor_offsets=torch.full((2, 12), 0.01),
    )
    action = SourceDelayedJointPositionActionCfg(
        entity_name="robot", actuator_names=JOINT_NAMES, scale=0.25
    ).build(env)
    action.process_actions(torch.ones(2, 12))
    action._switch[:] = torch.tensor([[0], [3]])
    action.apply_actions()
    torch.testing.assert_close(robot.target[0], torch.full((12,), 0.26))
    torch.testing.assert_close(robot.target[1], torch.full((12,), 0.01))
    for _ in range(3):
        action.apply_actions()
    torch.testing.assert_close(robot.target, torch.full((2, 12), 0.26))
    action.reset(torch.arange(2))
    assert action.raw_action.all()


def test_startup_randomization_records_source_privileged_layout():
    from src.tasks.go2.standup.mdp import randomize_source_physics, JOINT_NAMES

    n = 16
    model = SimpleNamespace(
        geom_friction=torch.ones(n, 4, 3),
        geom_solref=torch.ones(n, 4, 2),
        body_mass=torch.full((n, 13), 2.0),
        body_inertia=torch.ones(n, 13, 3),
        body_ipos=torch.zeros(n, 13, 3),
        dof_armature=torch.zeros(n, 18),
        dof_frictionloss=torch.zeros(n, 18),
        dof_damping=torch.zeros(n, 18),
    )
    actuator = SimpleNamespace(stiffness=torch.zeros(n, 12), damping=torch.zeros(n, 12))

    class Robot:
        indexing = SimpleNamespace(
            geom_ids=torch.arange(4),
            body_ids=torch.arange(13),
            joint_v_adr=torch.arange(6, 18),
            root_body_id=0,
        )
        actuators = [actuator]

        def find_bodies(self, pattern):
            return [0], ["base_link"]

        def find_joints(self, patterns, preserve_order):
            return list(range(12)), JOINT_NAMES

    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        sim=SimpleNamespace(model=model),
        scene={"robot": Robot()},
    )
    randomize_source_physics(env, None)
    domain = env._standup_v2_domain
    assert domain.shape == (n, 34)
    assert (domain[:, 0] == 0).all()
    torch.testing.assert_close(domain[:, -1], domain[:, -2])
    torch.testing.assert_close(actuator.stiffness, 40 * domain[:, 5:17])
    torch.testing.assert_close(actuator.damping, domain[:, 17:29])
    assert model.body_mass[:, 0].min() >= 1.0 and model.body_mass[:, 0].max() <= 4.0
    assert (
        model.geom_friction[:, :, 0].min() >= 0.2
        and model.geom_friction[:, :, 0].max() <= 1.2
    )
    assert env._standup_v2_motor_offsets.abs().max() <= 0.035


def test_source_timeout_is_strict_and_symmetry_is_involutive():
    from src.tasks.go2.standup.mdp import source_timeout
    from src.tasks.go2.standup.symmetry import mirror
    from src.tasks.go2.standup.source_config import Go2_Handstand_PPO_Yu as Ppo

    assert source_timeout(
        SimpleNamespace(
            episode_length_buf=torch.tensor([1000, 1001]), max_episode_length=1000
        )
    ).tolist() == [False, True]
    obs = torch.randn(5, 45)
    torch.testing.assert_close(
        mirror(
            mirror(obs, Ppo.algorithm.obs_permutation), Ppo.algorithm.obs_permutation
        ),
        obs,
    )


def test_export_action_key_and_calf_contact_geometry():
    import re
    from src.tasks.go2.standup.env_cfg import standup_env_cfg

    cfg = standup_env_cfg()
    assert "joint_pos" in cfg.actions, "Native export metadata requires joint_pos"
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    for leg in ("FL", "FR", "RL", "RR"):
        for part in ("thigh", "calf"):
            primary = sensors[f"body_{leg}_{part}"].primary
            assert primary.mode == "geom", (
                "Body calf contact includes its attached foot sphere"
            )
            assert re.fullmatch(primary.pattern, f"{leg}_{part}1_collision")
            assert not re.fullmatch(primary.pattern, f"{leg}_foot_collision")


def test_contact_adapter_sums_geom_vectors_before_source_body_reward():
    from src.tasks.go2.standup.mdp import contact_forces

    names = [f"foot_{leg}" for leg in ("FL", "FR", "RL", "RR")]
    names += [
        f"body_{leg}_{part}"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf")
    ]
    names += ["base_contact"]
    scene = {
        name: SimpleNamespace(data=SimpleNamespace(force=torch.zeros(1, 1, 3)))
        for name in names
    }
    scene["body_FL_calf"].data.force = torch.tensor(
        [[[0.0, 0.0, 0.06], [0.0, 0.0, 0.06]]]
    )
    scene["body_FR_calf"].data.force = torch.tensor(
        [[[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]]
    )
    forces = contact_forces(SimpleNamespace(scene=scene))
    assert forces.shape == (1, 13, 3)
    reward = load("source_rewards").SourceRewards()
    reward.contact_forces = forces
    reward.penalised_contact_indices = list(range(4, 12))
    torch.testing.assert_close(reward._reward_collision(), torch.tensor([1.0]))


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

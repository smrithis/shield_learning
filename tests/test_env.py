import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from shield_learning import ShieldInterceptionEnv
from shield_learning.scenarios import throw_for_crossing


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ShieldInterceptionEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_gym_contract(self):
        check_env(self.env, skip_render_check=True)

    def test_seeded_rollout_replay(self):
        traces = []
        for _ in range(2):
            obs, info = self.env.reset(seed=42)
            trace = [obs]
            for _ in range(3):
                action = self.env.baseline_action()
                obs, reward, terminated, truncated, info = self.env.step(action)
                trace.append((obs, reward, terminated, truncated, info))
            traces.append(trace)
        def comparable(trace):
            import json
            return json.dumps(trace, default=lambda x: x.tolist(), sort_keys=True)
        self.assertEqual(comparable(traces[0]), comparable(traces[1]))
        self.assertGreater(traces[0][-1][-1]['max_joint_displacement_rad'], 0.01)
        obs_a, _ = self.env.reset(seed=1)
        obs_b, _ = self.env.reset(seed=2)
        self.assertFalse(np.array_equal(obs_a['projectile_mean'], obs_b['projectile_mean']))

    def test_invalid_actions_and_options(self):
        self.env.reset(seed=1)
        for action in [-1, self.env.action_space.n, 1.5, True, np.bool_(True), np.array([1])]:
            with self.assertRaises(ValueError):
                self.env.step(action)
        self.env.step(np.int64(1))
        with self.assertRaises(RuntimeError):
            self.env.reset(options={'unknown': 1})
        with self.assertRaises(RuntimeError):
            self.env.step(1)
        self.env.reset(seed=1)

    def test_ground_termination(self):
        self.env.reset(seed=0, options={'randomize': False, 'throw': {'z0': 0.0, 'launch_angle_deg': -10}})
        _, reward, terminated, truncated, info = self.env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'ground')
        self.assertFalse(info['is_success'])
        self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))
        with self.assertRaises(RuntimeError):
            self.env.step(1)

    def test_timeout_and_close(self):
        env = ShieldInterceptionEnv(max_ticks=3, substeps=10)
        try:
            with self.assertRaises(RuntimeError):
                env.step(1)
            env.reset(seed=0, options={'randomize': False})
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertAlmostEqual(info['elapsed_s'], .012)
            process = env._process
        finally:
            env.close()
            env.close()
        self.assertIsNotNone(process.poll())
        with self.assertRaises(RuntimeError):
            env.reset()

    def test_geometry_verification_and_overlap(self):
        report = self.env.verify()
        self.assertTrue(report['bridge_integration_pass'])
        # Regression for the world-vs-plane mix-up: raw MuJoCo world FK minus plane
        # targets gave a 0.800 m median. Corrected comparison is at rounding level.
        self.assertAlmostEqual(report['legacy_world_minus_plane_median_xyz_error_m'], .8, places=6)
        self.assertLess(report['stored_goal_fk_median_xyz_error_m'], 1e-7)
        self.assertLess(report['world_from_plane_vs_run_ik_max_error'], 1e-7)
        self.assertLess(report['urdf_shield_center_max_error_m'], 1e-7)
        self.assertLess(report['urdf_shield_rotation_max_element_error'], 1e-7)
        # Remaining genuine tree outliers; this changes only if the tree does.
        self.assertEqual(report['stored_goal_fk_count_over_m'], {'0.001': 34, '0.01': 18, '0.05': 7, '0.1': 4})
        np.testing.assert_allclose(report['home_shield_center_odom'], [0.792059, 0.212231, 0.9615], atol=1e-6)
        errors = self.env.backend_metadata['goal_fk_error_m']
        self.assertEqual(sum(e > 1e-3 for e in errors), 34)
        x, y, z = report['home_shield_center_odom']
        self.env.reset(seed=0, options={'randomize': False, 'throw': {'x0': x, 'y0': y, 'z0': z}})
        _, reward, terminated, truncated, info = self.env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'interception')
        self.assertEqual(info['elapsed_s'], 0.)
        self.assertEqual(reward, 10.)
        self.assertFalse(info['reward_is_physical_contact'])

    def test_shield_frame_matches_odom(self):
        # Static check at reset: a ball 10 cm in front of the corrected home shield
        # centre is outside, one at the centre inside; the legacy centre is far away.
        x, y, z = 0.792059, 0.212231, 0.9615
        _, info = self.env.reset(seed=0, options={'randomize': False, 'throw': {'x0': x + 0.1, 'y0': y, 'z0': z}})
        self.assertGreater(info['minimum_surface_distance_m'], 0.04)
        self.assertLess(info['minimum_surface_distance_m'], 0.1)
        _, info = self.env.reset(seed=0, options={'randomize': False, 'throw': {'x0': x, 'y0': y, 'z0': z}})
        self.assertLess(info['minimum_surface_distance_m'], 0.)
        _, info = self.env.reset(seed=0, options={'randomize': False, 'throw': {'x0': 1.482882, 'y0': -0.202949, 'z0': z}})
        self.assertGreater(info['minimum_surface_distance_m'], 0.5)
        self.env.reset(seed=0)

    def test_swept_check_agrees_with_sampling(self):
        self.env.reset(seed=3)
        while True:
            _, _, terminated, truncated, info = self.env.step(self.env.baseline_action())
            if terminated or truncated:
                break
        swept = self.env.swept_check(subdivisions=20)
        self.assertEqual(swept['swept_sample_interval_s'], .004 / 20)
        self.assertLessEqual(swept['swept_min_surface_distance_m'], info['minimum_surface_distance_m'] + 1e-6)
        self.assertLess(swept['evaluator_vs_mujoco_max_outside_disagreement_m'], 1e-6)

    def test_controlled_interception(self):
        for moving in (True, False):
            result = self.env.controlled_interception(44, moving=moving)
            swept = result['episode_swept']
            self.assertEqual(result['termination_reason'], 'interception')
            self.assertTrue(result['is_success'])
            self.assertLess(result['goal_fk_error_m'], 1e-6)
            self.assertEqual(result['probe_replay_max_joint_error_rad'], 0.)
            self.assertLess(result['aim_error_m'], 1e-9)
            self.assertLessEqual(swept['swept_sample_interval_s'], 4e-5)
            self.assertLess(swept['swept_max_relative_motion_between_samples_m'], 1e-3)
            # Fine check finds contact no later than the 4 ms sample, and within one tick.
            first = swept['swept_first_contact_time_s']
            self.assertIsNotNone(first)
            self.assertLessEqual(first, result['sampled_termination_time_s'])
            self.assertLess(result['sampled_termination_time_s'] - first, .004)
            self.assertLess(swept['evaluator_vs_mujoco_max_outside_disagreement_m'], 1e-6)
            if moving:
                self.assertGreater(result['shield_speed_at_designed_contact_m_s'], .5)
                self.assertGreater(result['throw']['x0'], 2.)
            else:
                self.assertEqual(result['shield_speed_at_designed_contact_m_s'], 0.)
        with self.assertRaises(RuntimeError):
            self.env.step(1)

    def test_goal_mask(self):
        mask = self.env.action_masks()
        errors = np.asarray(self.env.backend_metadata['goal_fk_error_m'])
        np.testing.assert_array_equal(np.flatnonzero(~mask), [0, 16, 101, 117, 131, 132, 162])
        np.testing.assert_array_equal(~mask, errors > .05)
        self.assertEqual(self.env.action_space.n, 163)  # Tree and indices unchanged.
        _, info = self.env.reset(seed=0, options={'randomize': False})
        self.assertEqual(info['action_mask'], mask.tolist())
        for action in np.flatnonzero(~mask):
            with self.assertRaises(ValueError):
                self.env.step(int(action))
        self.assertTrue(mask[self.env.baseline_action()])

    def test_mask_changes_baseline_selection(self):
        # Throw crossing node 1's stored target (whose shield actually goes 0.79 m away).
        R = self.env.backend_metadata['base_from_plane_rotation']
        t = self.env.backend_metadata['base_from_plane_translation']
        y, z = self.env.backend_metadata['goal_task_positions_m'][0]
        throw, _ = throw_for_crossing(R, t, y, z, 6.5, 10.0, 0.0, 4.0)
        outcomes = {}
        unmasked = ShieldInterceptionEnv(apply_goal_mask=False)
        try:
            self.assertTrue(unmasked.action_masks().all())
            for name, env in (('masked', self.env), ('unmasked', unmasked)):
                for policy in ('planner', 'discrete'):
                    env.reset(seed=0, options={'randomize': False, 'throw': throw})
                    while True:
                        step = env.step_planner() if policy == 'planner' else env.step(env.baseline_action())
                        if step[2] or step[3]:
                            break
                    outcomes[name, policy] = (step[4], env.episode_diagnostics())
        finally:
            unmasked.close()
        for policy in ('planner', 'discrete'):
            info, diagnostics = outcomes['unmasked', policy]
            self.assertEqual(info['selected_goal_node_index'], 1)
            self.assertTrue(info['selected_goal_flagged'])
            self.assertFalse(info['is_success'])
            self.assertEqual(diagnostics['failure_cause'], 'goal_fk_error')
            info, diagnostics = outcomes['masked', policy]
            self.assertFalse(info['selected_goal_flagged'])
            self.assertTrue(info['is_success'])
            self.assertIsNone(diagnostics['failure_cause'])

    def test_launch_time_and_diagnostics(self):
        throw = {'x0': 4.0, 'y0': 0.0, 'z0': 1.1, 'speed': 6.5, 'launch_angle_deg': 8.0, 'side_angle_deg': 0.0}
        obs0, _ = self.env.reset(seed=0, options={'randomize': False, 'throw': throw})
        obs1, _ = self.env.reset(seed=0, options={'randomize': False, 'throw': throw, 'launch_time_s': -0.2})
        v = obs0['projectile_mean'][3:]
        np.testing.assert_allclose(obs1['projectile_mean'][:3],
                                   obs0['projectile_mean'][:3] + 0.2 * v + [0, 0, -0.5 * 9.81 * 0.04], atol=1e-12)
        for bad in (0.1, -2.0):
            with self.assertRaises(RuntimeError):
                self.env.reset(options={'randomize': False, 'launch_time_s': bad})
        # Beyond the goal grid: the ball crosses 1.3 m to the side.
        R = self.env.backend_metadata['base_from_plane_rotation']
        t = self.env.backend_metadata['base_from_plane_translation']
        far, _ = throw_for_crossing(R, t, 1.3, 1.1, 6.5, 8.0, 0.0, 4.0)
        self.env.reset(seed=0, options={'randomize': False, 'throw': far})
        while not self.env.step(self.env.baseline_action())[2]:
            pass
        diagnostics = self.env.episode_diagnostics()
        self.assertAlmostEqual(diagnostics['crossing_yz_m'][0], 1.3, places=6)  # rounded planner R
        self.assertAlmostEqual(diagnostics['crossing_yz_m'][1], 1.1, places=6)
        self.assertEqual(diagnostics['failure_cause'], 'outside_goal_coverage')
        self.assertGreater(self.env.swept_check(20)['swept_min_surface_distance_m'], 0)

    def test_constructor_validation(self):
        for config in [{'substeps': 0}, {'max_ticks': 1.5}, {'motion_weight': -1}, {'timeout': float('nan')},
                       {'goal_mask_fk_error_m': 0.0}, {'goal_mask_fk_error_m': float('inf')}]:
            with self.assertRaises(ValueError):
                ShieldInterceptionEnv(**config)


if __name__ == '__main__':
    unittest.main()

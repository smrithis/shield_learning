// C ABI only. All trajectory generation and limits come from the existing library.
#include <belief_planner/existing_ruckig_controller.hpp>
#include <algorithm>
#include <exception>

struct Controller {
    shield_controller::ExistingRuckigController core;
    ruckig::InputParameter<6> input;
    ruckig::OutputParameter<6> output;
};

extern "C" {
void* shield_create(const double* q) {
    try {
        auto* c = new Controller;
        c->core.configurePlanningLimits(c->input);
        std::copy_n(q, 6, c->input.current_position.begin());
        std::copy_n(q, 6, c->input.target_position.begin());
        return c;
    } catch (...) { return nullptr; }
}

void shield_destroy(void* ptr) { delete static_cast<Controller*>(ptr); }

void shield_command(void* ptr, const double* q, const double* v,
                    const double* a, double duration) {
    auto& c = *static_cast<Controller*>(ptr);
    shield_controller::PlannerCommand cmd;
    std::copy_n(q, 3, cmd.position.begin());
    std::copy_n(q + 3, 3, cmd.position2.begin());
    std::copy_n(v, 3, cmd.velocity.begin());
    std::copy_n(a, 3, cmd.acceleration.begin());
    cmd.duration = duration;
    c.core.applyPlannerCommand(cmd, c.input);
}

int shield_tick(void* ptr, const double* measured_q, double* velocity) {
    try {
        auto& c = *static_cast<Controller*>(ptr);
        shield_controller::RobotJointState state;
        std::copy_n(measured_q, 6, state.position.begin());
        c.core.applyRobotJointState(state, c.input);
        const auto result = c.core.update(c.input, c.output);
        if (static_cast<int>(result) < 0) return static_cast<int>(result);
        const auto command = c.core.makeOutput(c.output, result);
        std::copy(command.joint_velocity.begin(), command.joint_velocity.end(), velocity);
        c.output.pass_to_input(c.input);
        // The core interprets this as remaining time; consume simulated time.
        if (c.input.minimum_duration) {
            const double left = *c.input.minimum_duration - 0.004;
            if (left > 0) c.input.minimum_duration = left;
            else c.input.minimum_duration.reset();
        }
        return static_cast<int>(result);
    } catch (...) { return -999; }
}
}

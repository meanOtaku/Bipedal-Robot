# Empirical Comparison of FSM and CPG Gait Generation for a 12-DOF MuJoCo Biped

**Authors:** Vaibhav Bisht, Rahul Chandra Padamuttam, Kanktekar Srinidhi, Kaleru Ankitha

## Abstract

This paper presents an empirical comparison between two bipedal gait-generation strategies implemented for a 12-degree-of-freedom anthropomorphic robot in MuJoCo: a discrete finite-state machine (FSM) controller and a phase-driven central pattern generator (CPG) controller. The FSM baseline produces walking through timed transitions between joint-space keyframes, while the CPG controller replaces the discrete swing-leg keyframes with a continuous phase trajectory while retaining the same balance shell, proportional-derivative joint control, and stance-transfer logic. Telemetry from the controllers is used to compare joint-space coordination, sagittal foot trajectories, actuator torque jerk, stance-phase parasitic work, and torso posture. The results show that the CPG representation regularizes the swing-leg path into a smoother phase manifold and produces cleaner hip-knee coordination than the FSM baseline. However, the experiments also reveal that smooth geometric motion does not automatically imply lower mechanical cost: unfiltered attitude feedback and aggressive ankle-compliance scheduling can introduce high-frequency torque variation and additional negative braking work. The study concludes that a practical bipedal controller should combine FSM-level supervisory safety, CPG-level phase continuity, filtered postural reflexes, and bounded impedance scheduling.

**Keywords:** bipedal locomotion, finite-state machine, central pattern generator, MuJoCo, humanoid robot, actuator jerk, parasitic work, gait control

## 1. Introduction

Bipedal locomotion is difficult because the robot must continuously coordinate unstable floating-base dynamics, unilateral foot-ground contacts, actuator limits, and changing support phases. A walking controller must therefore solve two problems at the same time. First, it must generate feasible leg trajectories that move the swing foot forward without collision. Second, it must maintain a stable center of mass and recover from disturbances while the support polygon changes every step.

The project studied here contains two implemented approaches to this problem. The first is a finite-state machine controller in `FSM vs CPG/Walk with FSM.py`. This controller decomposes walking into explicit states such as standing, leaning, lifting the right leg, planting the right leg, shifting weight, lifting the left leg, and returning to the next support phase. Each state tracks a dictionary of desired joint angles using a smooth temporal interpolation and a joint-space PD controller.

The second approach is implemented in `FSM vs CPG/Walk with CPG.py`. It preserves the broad state-machine structure for balance and stance transfer, but replaces the discrete lift-and-plant swing keyframes with a phase-indexed CPG swing trajectory. Instead of jumping between manually authored swing poses, the swing leg is generated from a continuous phase variable. This makes the gait more suitable for speed scaling and cyclic motion analysis.

The central research question is not simply whether the CPG looks smoother. The deeper question is whether geometric smoothness improves the mechanical behavior of the whole robot once feedback, compliance, contact, and actuator effort are included. The results show a useful trade-off: the CPG improves the shape of the swing trajectory, but the current reflex and ankle-compliance choices can increase hidden actuator and energy costs.

## 2. Related Work

FSM-based gait controllers are common in early humanoid and legged-robot prototypes because they are transparent, easy to debug, and naturally aligned with contact events. A designer can explicitly define when the robot should shift weight, lift a foot, plant a foot, and settle. This makes FSM controllers practical for simulation bring-up, but they often create discontinuities at state boundaries because the robot is commanded to leave one local motion primitive and enter another.

CPG-based controllers are inspired by biological rhythmic motion. They replace hard state-to-state swing commands with phase-driven oscillatory trajectories. A phase variable can progress faster or slower depending on desired walking speed, while the generated trajectory remains cyclic and smooth. This makes CPGs attractive for repetitive locomotion, especially when combined with feedback that adapts the phase or amplitude to the robot state.

Modern biped control often combines both ideas. A high-level state machine supervises contact logic and safety, while continuous trajectory generators shape the swing leg and impedance controllers regulate stance. The experiments in this project follow that hybrid direction: the CPG controller is not a pure oscillator-only walker, but rather a CPG swing generator embedded inside a stabilizing FSM-like walking sequence.

## 3. Robot Model and Simulation Setup

The experiments use a MuJoCo biped model defined in `FSM vs CPG/biped.xml`. The robot has a floating base and 12 actuated lower-body joints:

- left and right hip yaw
- left and right hip roll
- left and right hip pitch
- left and right knee pitch
- left and right ankle pitch
- left and right ankle roll

The MuJoCo simulation resolves rigid-body dynamics, joint constraints, gravity, ground contact, and actuator torques. Both controllers use the same joint list, the same basic PD actuation structure, and the same telemetry style. This makes the comparison meaningful because the primary difference is the gait-generation method rather than the robot model.

Telemetry is recorded from the simulated robot into CSV files. The logged signals include time, state, mode, torso position, torso velocity, torso roll-pitch-yaw attitude, torso angular velocity, and each joint's position, velocity, and torque. These telemetry streams are then analyzed by graph scripts in `FSM vs CPG/Graphs`.

## 4. FSM Controller

The FSM controller generates walking through discrete keyframe transitions. It defines named poses such as `sit`, `stand`, `lean_L_initial`, `lift_R`, `plant_R`, `center_R`, `lean_R`, `lift_L`, `plant_L`, `center_L`, and `lean_L`. The controller progresses through numbered walking states:

- sitting and standing initialization
- idle standing
- initial weight shift
- right-leg swing and plant
- right-support weight transfer
- left-leg swing and plant
- left-support weight transfer
- optional settling and fall handling

For each state transition, the reference posture is interpolated using the cubic smoothstep function:

```text
alpha(t) = 3(t / DeltaT)^2 - 2(t / DeltaT)^3
```

This interpolation removes abrupt position jumps inside a transition. However, the overall motion remains segmented because the next target pose and state duration are chosen discretely. In practice, the foot path and joint cyclograms can still show sharp vertices where the gait changes from one authored keyframe segment to the next.

The FSM approach has two main strengths. First, it is easy to understand and modify because every walking phase is visible as a named pose or state. Second, it is robust during early experimentation because the controller can be tuned phase by phase. Its weakness is that the gait is tightly coupled to state durations and authored joint targets. When speed, stride, or contact timing changes, the discrete pose chain can become mechanically harsh.

## 5. CPG Controller

The CPG controller keeps the broader walking sequence but replaces the swing-leg keyframes with a phase-driven trajectory function:

```python
cpg_swing_trajectory(phase, speed)
```

The phase variable ranges from 0 to 1 over the swing duration. Speed affects both the swing timing and the stride scale. The trajectory function generates hip, knee, and ankle targets for the swing leg. The knee reaches a configurable peak during clearance, while the hip and ankle move through smooth phase-dependent transitions.

The controller still uses FSM-like states for initialization, idle standing, lean transfer, swing, support shift, and settling. This is important: the implementation is best understood as a hybrid FSM-CPG controller. The FSM provides contact sequencing and safety structure, while the CPG provides smoother swing-leg geometry.

The CPG controller also adds active ankle balance. During walking states, it reads torso pitch and pitch rate from the simulated IMU state and applies a bounded ankle-pitch correction:

```text
pitch_correction = -0.1 * pitch - 0.02 * pitch_rate
```

The correction is clipped to a small interval before being added to the stance ankle target. This reflex improves posture regulation, but it also introduces an important research issue: if raw attitude feedback is injected directly into low-level targets, the actuator torque can become noisy even when the visible foot trajectory appears smooth.

## 6. Data Analysis Methods

The project includes several scripts that convert controller telemetry into diagnostic plots.

### 6.1 Joint-Space Cyclogram

`FSM vs CPG/Graphs/Fig 1 (2)/plot_fsm_vs_cpg_cyclogram.py` compares hip and knee coordination. It plots left hip pitch against left knee pitch after removing the initial transient. This reveals whether the gait forms a smooth repeatable loop or a segmented keyframe path.

### 6.2 Sagittal Foot Manifold

`FSM vs CPG/Graphs/Fig 1/plot_kinematic_manifold.py` reconstructs the right foot position relative to the torso and plots the sagittal path. The local coordinates are computed as:

```text
x_local = right_foot_x - torso_x
z_local = right_foot_z - torso_z
```

This makes the plot useful for future inverse-kinematics work because it examines the foot path relative to the robot body rather than only in global world coordinates.

### 6.3 Actuator Jerk Spectrum

`FSM vs CPG/Graphs/Fig 2/plot_spectral_jerk.py` analyzes the frequency content of actuator torque changes. It uses the left knee pitch torque and computes discrete actuator jerk as:

```text
jerk[n] = (torque[n] - torque[n - 1]) / dt
```

The script then applies a Fourier transform and plots the power spectral density. This reveals whether a controller creates high-frequency actuator activity that may not be obvious from the visible animation.

### 6.4 Parasitic Work

`FSM vs CPG/Graphs/Fig 3/plot_parasitic_work.py` computes ankle mechanical power from torque and angular velocity:

```text
power = torque * angular_velocity
```

It then isolates a late-stance window around the strongest negative power event and integrates only the negative portion. This produces an estimate of parasitic braking work: energy absorbed or wasted by the actuator while resisting motion.

### 6.5 Postural Attitude

`FSM vs CPG/Graphs/Fig 4/plot_postural_attitude.py` compares torso height and pitch over a 10-second window. This verifies whether differences in actuator cost are caused by global instability or by more subtle internal control behavior.

## 7. Results

### 7.1 Kinematic Regularization

The kinematic plots show that the FSM gait contains sharper transitions in the sagittal foot path and in the hip-knee cyclogram. This is expected because the FSM walks by moving from one manually defined pose to the next. Even with smoothstep interpolation inside each transition, the target path remains piecewise.

The CPG trajectory produces a more continuous swing pattern. Because the leg is driven by a phase variable, the swing motion forms a cleaner cyclic path. This is useful for walking-speed changes and future inverse-kinematics learning because the model can learn from a consistent phase-conditioned relationship between body-relative foot placement and joint angles.

### 7.2 Spectral Jerk

The actuator-jerk analysis shows why visual smoothness is not enough. A controller can produce a smooth-looking foot path while still commanding high-frequency torque changes internally. The extracted research report identifies a large high-frequency jerk penalty in the CPG case when raw 100 Hz attitude feedback is superimposed on the compliant phase-driven motion.

This result does not mean that CPGs are inherently worse than FSMs. Rather, it shows that the current CPG implementation needs better feedback conditioning. The phase trajectory smooths the desired swing geometry, but the ankle reflex can reintroduce high-frequency corrections at the actuator level. A low-pass filter, observer, or impedance-aware reflex boundary would likely reduce this effect.

### 7.3 Parasitic Work

The parasitic-work analysis indicates that late-stance ankle compliance must be scheduled carefully. The extracted report gives a representative comparison of 1.73 J of negative braking work for the FSM case and 2.68 J for the CPG case, corresponding to roughly 54.9% more parasitic work in that trial.

The likely cause is excessive compliance during stance transfer. When ankle stiffness is lowered too aggressively, the robot can yield downward faster than intended. Other joints must then absorb energy to arrest the falling torso and maintain posture. This creates wasted negative work even though the gait may appear stable.

### 7.4 Postural Equivalence

The posture plots show that both controllers can keep the global torso motion bounded. The center-of-mass height remains within a narrow vertical band, and pitch oscillation remains controlled. This is an important observation because it means the jerk and parasitic-work differences are not simply caused by one controller falling or becoming globally unstable. They are internal cost differences inside otherwise successful walking trials.

## 8. Discussion

The experiments suggest that FSM and CPG controllers should not be viewed as simple competitors where one is always better. They solve different parts of the locomotion problem well.

The FSM controller is transparent and reliable for phase sequencing. It is good for early development because each walking state can be inspected and tuned directly. Its limitation is that it can generate piecewise motion with non-smooth spatial transitions.

The CPG controller improves swing-leg continuity and gives a better foundation for rhythmic walking. It also gives a natural input structure for future inverse-kinematics work: phase, body-relative foot position, and desired velocity can be mapped to joint angles. However, the CPG controller also exposes the danger of combining smooth trajectories with unfiltered reflexes and overly soft stance compliance.

The most promising architecture is therefore hybrid:

- FSM supervision for contact order, startup, stopping, fall handling, and safety gates
- CPG swing generation for smooth cyclic foot motion
- filtered IMU feedback for posture correction
- bounded impedance scheduling to prevent stance collapse
- telemetry-driven optimization using actuator jerk, parasitic work, and foot-placement consistency

## 9. Implementation Limitations

The current CPG implementation uses cubic smoothstep-style interpolation. This gives smooth position and zero endpoint velocity within each segment, but strict acceleration-continuous C2 behavior across all phase boundaries would require a quintic minimum-jerk polynomial, a periodic spline, or explicit acceleration matching. Future versions should upgrade the phase basis before claiming full C2 continuity at every boundary.

The current analysis also depends on telemetry generated from simulation. MuJoCo is highly useful for debugging control logic, but hardware transfer would require actuator bandwidth limits, sensor delay, encoder quantization, foot-contact uncertainty, and motor thermal limits to be modeled or measured.

Finally, the actuator-jerk and parasitic-work results should be evaluated over multiple walking speeds and repeated trials. The project already supports speed-scaled CPG walking, so the next step is to batch-run several speeds and produce tables of mean jerk, peak jerk, negative work, stride length, and torso pitch variation.

## 10. Future Work

Future work should focus on converting the current comparison into a repeatable optimization pipeline.

First, the foot-placement telemetry should be saved relative to the pelvis or stance foot so it can become an input dataset for inverse kinematics. The desired output would be the 12 joint angles required to reproduce the body-relative foot pose.

Second, the CPG trajectory should be upgraded from cubic phase shaping to a periodic minimum-jerk or spline-based generator. This would make the smoothness claim stronger and reduce boundary artifacts.

Third, the ankle reflex should be filtered. A zero-lag offline filter is useful for analysis, but a deployable controller should use a causal low-pass filter or observer that does not require future samples.

Fourth, impedance scheduling should be constrained by stance phase. The trailing ankle can be compliant enough to allow rollover, but not so compliant that the pelvis drops and forces the hip and knee to absorb unnecessary negative work.

## 11. Conclusion

This project demonstrates that CPG-based gait generation can improve the geometric quality of a bipedal walking trajectory compared with a purely keyframe-driven FSM. The phase-driven swing leg produces smoother cyclic motion and better body-relative foot-path structure. However, the results also show that smooth kinematics alone are not sufficient for efficient humanoid locomotion. When raw posture feedback and aggressive compliance are layered onto the controller, the robot can pay hidden costs in actuator jerk and parasitic braking work.

The best direction for the robot is a hybrid architecture: keep the FSM as a safety and contact supervisor, use the CPG as the swing trajectory generator, filter reflex feedback before it reaches the actuators, and tune stance impedance using energy and jerk metrics rather than visual stability alone.

## References

[1] Z. Xie, P. Gergondet, F. Kanehiro, et al., "Learning bipedal walking for humanoids with current feedback," IEEE Access, vol. 11, pp. 82013-82023, 2023.

[2] O. Aydogmus and M. Yilmaz, "Comparative analysis of reinforcement learning algorithms for bipedal robot locomotion," IEEE Access, vol. 12, pp. 7490-7499, 2023.

[3] S. Akki and T. Chen, "Benchmarking model predictive control and reinforcement learning based control for legged robot locomotion in MuJoCo simulation," IEEE Access, 2025.

[4] A. D. Ames, K. Galloway, K. Sreenath, and J. W. Grizzle, "Rapidly exponentially stabilizing control Lyapunov functions and hybrid zero dynamics," IEEE Transactions on Automatic Control, vol. 30, no. 4, pp. 876-891, 2014.

[5] L. C. P. Camacho, "An autonomous bipedal walking robot for online reinforcement learning," Dissertation, 2024.

[6] A. J. Ijspeert, "Central pattern generators for locomotion control in animals and robots: A review," Neural Networks, vol. 21, no. 4, pp. 642-653, 2008.

[7] A. Kumar, Z. Li, J. Zeng, D. Pathak, K. Sreenath, and J. Malik, "Adapting rapid motor adaptation for bipedal robots," in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022, pp. 1161-1168.


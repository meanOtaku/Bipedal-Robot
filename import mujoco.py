import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the biped robot walking controller.")
    gait_group = parser.add_mutually_exclusive_group()
    gait_group.add_argument("--fsm", action="store_true", help="Run the finite-state-machine gait controller.")
    gait_group.add_argument("--cpg", action="store_true", help="Run the CPG swing-trajectory gait controller.")
    parser.add_argument("--speed", type=float, default=1.0, help="CPG walking speed scale. Default: 1.0")
    return parser.parse_args()


args = parse_args()

if args.cpg:
    from biped_robot.cpg_controller import run

    run(speed=args.speed)
else:
    import biped_robot.controller

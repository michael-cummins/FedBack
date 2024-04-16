import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with command line parsing.")
    parser.add_argument("--avg", action='store_true', default=False, help="Enable avg (default: False)")
    parser.add_argument("--prox", action='store_true', default=False, help="Enable prox (default: False)")
    parser.add_argument("--admm", action='store_true', default=False, help="Enable admm (default: False)")
    parser.add_argument("--forward", action='store_true', default=False, help="Enable admm (default: False)")
    parser.add_argument("--back", action='store_true', default=False, help="Enable admm (default: False)")
    parser.add_argument("--rate", type=int, required=True, help="Set the rate in percentage")

    args = parser.parse_args()

    # Check for exclusive flags
    if sum([args.avg, args.prox, args.admm, args.forward, args.back]) != 1:
        parser.error("Exactly one of --avg, --prox, or --admm must be set to True.")
    experiments = ['cifar', 'mnist']
    for experiment in experiments:
        if args.forward or args.back:
            if args.forward: fname='forward'
            elif args.back: fname = 'back'
            file_path = f'event_subs/{experiment}/{fname}_{args.rate}.sub'
            with open(file_path, 'w') as f:
                f.write(f"executable = /usr/bin/python3\n")
                f.write(f"arguments = main.py --{experiment} --{fname} --rate {float(args.rate)/100}\n")
                f.write(f'error = errs/{experiment}/{fname}_{args.rate}.err\n')
                f.write(f'output = outs/{experiment}/{fname}_{args.rate}.out\n')
                f.write(f'log = logs/{experiment}/{fname}_{args.rate}.log\n')
                f.write(f'request_memory = 64096\nrequest_cpus = 1\nrequest_gpus = 1\nqueue')
        else:
            if args.avg: fname = 'avg'
            elif args.prox: fname = 'prox'
            elif args.admm: fname = 'admm'
            file_path = f'subs/{experiment}/'+fname+f'_{args.rate}.sub'
            with open(file_path, 'w') as f:
                f.write(f"executable = /usr/bin/python3\n")
                f.write(f"arguments = main.py --{experiment} --{fname} --rate {float(args.rate)/100}\n")
                f.write(f'error = errs/{experiment}/{fname}_{args.rate}.err\n')
                f.write(f'output = outs/{experiment}/{fname}_{args.rate}.out\n')
                f.write(f'log = logs/{experiment}/{fname}_{args.rate}.log\n')
                f.write(f'request_memory = 64096\nrequest_cpus = 1\nrequest_gpus = 1\nqueue')
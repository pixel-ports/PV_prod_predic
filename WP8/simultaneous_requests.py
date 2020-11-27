import time
import matplotlib.pyplot as plt
import argparse
from subprocess import Popen

def parallel_works(nb_processes, api_path):
    # command = ["xargs", "-I", "%", "-P", "10000", "curl http://localhost:5000", "< <(printf '%s\n' {1..3})"]
    command = "xargs -I % -P 1000 curl 'http://localhost:5000"+api_path+"' < <(printf '%s\n' {1.."+str(nb_processes)+"})"
    begin_time = time.time()
    worker_process = Popen(["bash", "-c", command])
    worker_process.wait()
    # subprocess.run(["bash", "-c", command], shell=True, check=True)
    return time.time() - begin_time


def main(min_processes, max_processes, step_processes, api_path):
    x, y = [], []
    for nb_processes in range(min_processes, max_processes, step_processes):
        print("Test with %d processes." % nb_processes)
        total_time = parallel_works(nb_processes, api_path)
        x.append(nb_processes)
        y.append(round(total_time, 2))
        print("total_time ==> %f" % y[-1])

    print(x)
    print(y)

    plt.xlabel("Number of process")
    plt.ylabel("Time (s)")
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process executable options.")
    parser.add_argument(
        "--min_processes",
        type=int,
        help="Max number of process to simultaneously start.",
    )
    parser.add_argument(
        "--max_processes",
        type=int,
        help="Max number of process to simultaneously start.",
    )
    parser.add_argument(
        "--step_processes", type=int, help="Step number of processes to increment."
    )
    parser.add_argument(
        "api_path", type=str, help="Path to call in the running api: e.g. / or /loaded_inference/"
    )
    args = parser.parse_args()

    main(args.min_processes, args.max_processes, args.step_processes, args.api_path)
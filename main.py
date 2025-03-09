# Libraries
import time
import subprocess
import matplotlib.pyplot as plt


if __name__ == "__main__":
    python_file = "CreditCard.py"
    cython_file = "CreditCardCython.py"

    array_time = []
    loop_size = 100
    
    for _ in range(loop_size):
        # Python Run Time
        start_time = time.time()
        subprocess.run(["python3", python_file], capture_output=True, text=True)
        python_time = time.time() - start_time

        # Cython Run Time
        start_time = time.time()
        subprocess.run(["python3", cython_file], capture_output=True, text=True)
        cython_time = time.time() - start_time

        # Result of Python and Cython
        print(f"Python Time = {python_time}")
        print(f"Cython Time = {cython_time}")
        speedup = python_time / cython_time
        print(f"Cython is faster than Python => {speedup:.2f}x")
        array_time.append(round(speedup, 2))

    # Draw Circle Plot
    cython_faster = sum(1 for s in array_time if s > 1)
    python_faster = loop_size - cython_faster
    labels = ["Cython", "Python"]
    sizes = [cython_faster, python_faster]
    colors = ["green", "blue"]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, explode=(0.1, 0))
    plt.title("Cython is faster than Python in runtimes")
    plt.show()

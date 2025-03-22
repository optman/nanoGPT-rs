import random

def generate_2x1_addition_problems(n, file):
    for _ in range(n):
        num1 = random.randint(0, 9)
        num2 = random.randint(0, 9)
        file.write(f"{num1}+{num2}={num1+num2}\n")

def generate_2x2_addition_problems(n, file):
    for _ in range(n):
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        file.write(f"{num1}+{num2}={num1+num2}\n")

def generate_3x1_addition_problems(n, file):
    for _ in range(n):
        num1 = random.randint(0, 9)
        num2 = random.randint(0, 9)
        num3 = random.randint(0, 9)
        file.write(f"{num1}+{num2}+{num3}={num1+num2}+{num3}={num1+num2+num3}\n")

def generate_2x1_subtraction_problems(n, file):
    for _ in range(n):
        num1 = random.randint(1, 9)
        num2 = random.randint(0, num1)
        file.write(f"{num1}-{num2}={num1-num2}\n")

def generate_2x2_subtraction_problems(n, file):
    for _ in range(n):
        num1 = random.randint(10, 99)
        num2 = random.randint(0, num1)
        file.write(f"{num1}-{num2}={num1-num2}\n")

for file in ["input-sft.txt", "input-rl.txt"]:
    with open(file, "w") as f:
        generate_2x1_addition_problems(10, f)
        generate_2x2_addition_problems(1000, f)
        generate_3x1_addition_problems(100, f)
        generate_2x1_subtraction_problems(10, f)
        generate_2x2_subtraction_problems(1000, f)

for file in ["input-test.txt"]:
    with open(file, "w") as f:
        generate_2x1_addition_problems(10, f)
        generate_2x2_addition_problems(100, f)
        generate_3x1_addition_problems(100, f)
        generate_2x1_subtraction_problems(10, f)
        generate_2x2_subtraction_problems(100, f)
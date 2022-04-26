from matplotlib import pyplot as plt
import argparse
import sys


def gradient_function_element(argument, const_value, position, length):
    return pow(const_value, (position-1)/(length-1)) * argument * 2

def hessian_function_element(const_value, position, length):
    return pow(const_value, (position-1)/(length-1)) *  2

def function_element(argument, const_value, position, length):
    return pow(const_value, (position-1)/(length-1)) * argument * argument

def gradient_descent(arguments, const_value, learning_rate):
    list_len = len(arguments)
    stop_condition = list_len
    epsilon = 0.00001
    values = []
    keys = []
    iterator = 0
    while(stop_condition != 0):
        function_sum = 0
        stop_condition = list_len
        for i in range(0, list_len):
            fun_result = gradient_function_element(arguments[i], const_value, i+1, list_len)
            function_sum += function_element(arguments[i], const_value, i+1, list_len)
            difference = fun_result * learning_rate
            if(abs(difference)<epsilon):
                stop_condition -= 1
            arguments[i] -= difference
        iterator += 1
        values.append(function_sum)
        keys.append(iterator)

    return keys, values

def newton_algorithm(arguments, const_value, learning_rate):
    list_len = len(arguments)
    stop_condition = list_len
    epsilon = 0.00001
    values = []
    keys = []
    iterator = 0
    while(stop_condition != 0):
        function_sum = 0
        stop_condition = list_len
        for i in range(0, list_len):
            fun_result = gradient_function_element(arguments[i], const_value, i+1, list_len)/hessian_function_element(const_value, i+1, list_len)
            function_sum += function_element(arguments[i], const_value, i+1, list_len)
            difference = fun_result * learning_rate
            if(abs(difference)<epsilon):
                stop_condition -= 1
            arguments[i] -= difference
        iterator += 1
        values.append(function_sum)
        keys.append(iterator)

    return keys, values

def newton_algorithm_backtrack(arguments, const_value, learning_rate, adjustment):
    list_len = len(arguments)
    stop_condition = list_len
    epsilon = 0.00001
    values = []
    keys = []
    old_fun_sum = sys.maxsize
    iterator = 0
    while(stop_condition != 0):
        function_sum = 0
        stop_condition = list_len
        arguments_copy = [0]*list_len
        for i in range(0, list_len):
            arguments_copy[i] = arguments[i]
            fun_result = gradient_function_element(arguments[i], const_value, i+1, list_len)/hessian_function_element(const_value, i+1, list_len)
            function_sum += function_element(arguments[i], const_value, i+1, list_len)
            difference = fun_result * learning_rate
            arguments[i] -= difference
            if(abs(difference)<epsilon):
                stop_condition -= 1
        if function_sum > old_fun_sum:
            learning_rate *= adjustment
            arguments = arguments_copy
        else:
            old_fun_sum = function_sum
        iterator += 1
        values.append(function_sum)
        keys.append(iterator)

    return keys, values


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, help='Specify method you want use: "g" for gradient descent, "n" for Newton, \
                        "nb" for the one with backtracking and m for combination of all', required=True)
    parser.add_argument('-l', '--list', nargs='+', type=int, help='Input list of arguments for this function', required=True)
    parser.add_argument('-c', '--constant', nargs=1, type=int, help='Specify constant value you want to use in those algorithms', required=True)
    parser.add_argument('-r', '--rate_of_learn', nargs=1, type=float, help='Specify the learning rate algorithms should use', required=True)
    parser.add_argument('-b', '--backtracking', nargs=1, type=float, help='Specify the amount by which the learning rate will be adjusted')

    args = parser.parse_args(arguments[1:])
    if(args.method == 'n'):
        data = newton_algorithm(args.list, args.constant[0], args.rate_of_learn[0])
        mode = "Newton"
    elif(args.method == 'g'):
        data = gradient_descent(args.list, args.constant[0], args.rate_of_learn[0])
        mode = "Gradient"
    elif(args.method == 'nb'):
        if(args.backtracking is None):
            print("Specify learning rate adjustment! (-b option)")
        else:
            data = newton_algorithm_backtrack(args.list, args.constant[0], args.rate_of_learn[0], args.backtracking[0])
            mode = "Newton with backtracking"
    elif(args.method == 'm'):
        if(args.backtracking is None):
            print("Specify learning rate adjustment! (-b option)")
        else:
            arguments_list = []
            for item in args.list:
                arguments_list.append(item)
            arguments_copy = []
            for item in arguments_list:
                arguments_copy.append(item)
            data = newton_algorithm(arguments_list, args.constant[0], args.rate_of_learn[0])
            mode = 'Newton'
            data_g = gradient_descent(arguments_copy, args.constant[0], args.rate_of_learn[0])
            mode_g = 'Gradient'
            # data_n = newton_algorithm_backtrack(args.list, args.constant[0], args.rate_of_learn[0], args.backtracking[0])
            # mode_n = 'Newton with backtracking'

    else:
        print("Choose correct algorithm mode!")

    plt.plot(data[0], data[1], '-', label=mode)
    if (args.method == 'm'):
        plt.plot(data_g[0], data_g[1], '-', label=mode_g)
        # plt.plot(data_n[0], data_n[1], '-', label=mode_n)
        mode = "Polaczenie"
    title = f'Działanie algorytmu - stala: {args.constant[0]} krok: {args.rate_of_learn[0]}'
    plt.title(title)
    plt.xlabel('Ilość iteracji')
    plt.ylabel('Wartość funkcji')
    plt.legend()
    plt.savefig(mode+'.png')


if __name__ == "__main__":
    main(sys.argv)

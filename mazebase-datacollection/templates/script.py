import sys

def main():
    while True:
        command = sys.stdin.readline()
        command = command.split('\n')[0]
        if command == "hello":
            sys.stdout.write("You said hello!\n")
        elif command == "goodbye":
            sys.stdout.write("You said goodbye!\n")
        else:
            sys.stdout.write("Sorry, I didn't understand that.\n")
        sys.stdout.flush()

if __name__ == '__main__':
    main()

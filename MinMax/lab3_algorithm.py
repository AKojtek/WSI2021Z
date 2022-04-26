class MyGame:
    def __init__(self, boardSize = 3, enablePlayer = False, playerStarts = True):
        """
        Init accepts as a variable size equal to one of the sides. If none is given sets size as 3
        Additionaly init defines if game should be played AI vs AI or player vs AI.
        In case of a player vs AI game user can set if he wants to start first.
        """
        self.isMaxi = False
        try:
            if not boardSize.isdigit():
                raise ValueError('Size should be a number!')
            self.size = int(boardSize)
            self.board = [['-' for _ in range(self.size)] for _ in range(self.size)]
            self.space = self.size * self.size
            self.maxi = self.size*self.size + 5
            self.mini = -(self.size*self.size + 5)
            self.maxDepth = int((self.size*self.size)*(3/4))
        except ValueError as exp:
            print("One of the values was incorrect! -> {}".format(exp))
        try:
            if enablePlayer in ("True", "False") and playerStarts in ("True", "False"):
                self.enablePlayer = enablePlayer == "True"
                self.playerStarts = playerStarts == "True"
            else:
                raise ValueError('Type of game and who starts accepts only True or False!')
        except ValueError as exp:
            print("One of the values was incorrect! -> {}".format(exp))


    # Accepting data from user
    # ========================
    # ========================
    def checkIfPositionCorrect(self,position):
        if not position[0].isdigit() or not position[1].isdigit():
            print("Enter correct position!")
            return False
        x = int(position[0])
        y = int(position[1])
        if x >= 0 and x < self.size and y >= 0 and y < self.size:
            return True
        print("Given numbers doesn't fit a set board!")
        return False


    def checkIfFree(self, position):
        if self.board[position[0]][position[1]] == '-':
            return True
        print('Given position is already taken!')
        return False


    def changeBoard(self, position, player):
        if player:
            self.board[position[0]][position[1]] = 'o'
        else:
            self.board[position[0]][position[1]] = 'x'
        self.space -= 1


    def takePosition(self, position):
        if self.checkIfPositionCorrect(position):
            x = int(position[0])
            y = int(position[1])
            if self.checkIfFree((x,y)):
                self.changeBoard((x,y),self.isMaxi)
                self.isMaxi = not self.isMaxi
                return True
            else:
                return False
        else:
            return False


    # Data output
    # =====================
    # =====================
    def drawBoard(self):
        print(f"-" * (3 * self.size) + "-")
        for row in self.board:
            print(*row, sep=" | ")
            print(f"-" * (3 * self.size) + "-")


    def drawInfo(self):
        self.drawBoard()
        if self.isMaxi:
            print("Now is player o turn")
        else:
            print("Now is player x turn")


    def finalInfo(self):
        print("Game over!")
        score = self.evaluateBoard()
        if score == self.maxi:
            print("Player 'o' is the winner")
        elif score == self.mini:
            print("Player 'x' is the winner")
        else:
            print("It's a tie!")


    # Algorithm
    # =====================
    # =====================
    def minMax(self, isMax, depth):
        score = self.evaluateBoard()

        if score == self.maxi:
            return score - depth

        if score == self.mini:
            return score + depth

        if not self.isRoomLeft() or depth > self.maxDepth:
            return 0

        if isMax:
            best = -100
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == '-':
                        self.changeBoard((i,j), isMax)

                        posValue = self.minMax(False, depth + 1)
                        best = max(best, posValue)
                        self.board[i][j] = '-'
                        self.space += 1

            return best
        else:
            best = 100
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == '-':
                        self.changeBoard((i,j), isMax)
                        posValue = self.minMax(True, depth + 1)
                        best = min(best, posValue)
                        self.board[i][j] = '-'
                        self.space += 1

            return best


    def findBestMove(self, isMaxi, depth):
        if isMaxi:
            bestVal = -100
        else:
            bestVal = 100
        bestMove = (-1, -1)

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == '-':
                    self.changeBoard((i,j), isMaxi)
                    moveVal = self.minMax(not isMaxi, depth + 1)
                    moveVal += self.evaluatePosition((i,j))
                    self.board[i][j] = '-'
                    self.space += 1

                    if self.checkVal(isMaxi, moveVal, bestVal):
                        bestVal = moveVal
                        bestMove = (i,j)

        return bestMove


    def checkVal(self, isMaxi, value, bestValue):
        if isMaxi:
            if value > bestValue:
                return True
            else:
                return False
        else:
            if value < bestValue:
                return True
            else:
                return False

    # How the game works
    # =====================
    # =====================
    def playAI(self):
        self.changeBoard(self.findBestMove(self.isMaxi, 0), self.isMaxi)
        self.isMaxi = not self.isMaxi
        self.drawInfo()
        input("Press enter to show next turn")


    def playPlayer(self):
        while True:
            x = input("Where you want to put your sign? (y variable): ")
            y = input("Now x variable: ")
            if self.takePosition((x,y)):
                self.drawInfo()
                input("Press enter to show next turn")
                break


    def isRoomLeft(self):
        if self.space > 0:
            return True
        else:
            return False


    def playGame(self):
        self.drawInfo()
        score = self.evaluateBoard()
        while self.isRoomLeft() and (score != self.maxi and score != self.mini):
            if not self.enablePlayer:
                self.playAI()
            elif self.playerStarts != self.isMaxi:
                self.playPlayer()
            else:
                self.playAI()
            score = self.evaluateBoard()
        self.finalInfo()


    # Evaluate value of certain parts of program
    # ==========================================
    # ==========================================
    def evaluateBoard(self):
        """
        Funtion determines state of the current board
        If function returns:
        10  -> The winner of current board is 'o'
        -10 -> The winner of current board is 'x'
        0  -> There is no winner for current board
        """

        # Checks each row
        for row in self.board:
            if all(element == row[0] for element in row):
                if row[0] == 'o':
                    return self.maxi
                elif row[0] == 'x':
                    return self.mini

        # Checks each column
        for i in range(self.size):
            counter = 1
            for j in range(1, self.size):
                if self.board[0][i] == self.board[j][i]:
                    counter += 1
            if counter == self.size:
                if self.board[0][i] == 'o':
                    return self.maxi
                elif self.board[0][i] == 'x':
                    return self.mini

        # Check one of the diagonals
        counter = 1
        for i in range(1, self.size):
            if self.board[i][i] == self.board[0][0]:
                counter += 1
        if counter == self.size:
            if self.board[0][0] == 'o':
                return self.maxi
            elif self.board[0][0] == 'x':
                return self.mini

        # check second of the diagonals
        counter = 1
        for i in range(1, self.size):
            if self.board[0][self.size - 1] == self.board[i][self.size - 1 - i]:
                counter += 1
        if counter == self.size:
            if self.board[0][self.size - 1] == 'o':
                return self.maxi
            elif self.board[0][self.size - 1] == 'x':
                return self.mini

        return 0


    # decide what is the value of certain position based on number of possible win which can go through it
    def evaluatePosition(self, position):
        if self.isMaxi:
            multiplier = 1
        else:
            multiplier = -1
        if position[0] == position[1]:
            if position[0] == self.size // 2:
                return 4 * multiplier
            else:
                return 3 * multiplier
        elif position[0]  == self.size - 1 - position[1]:
            return 3 * multiplier
        else:
            return 2 * multiplier


def main():
    size = input("What is the lenght of you tictactoe board?: ")
    gameType = input("Is it game AIvsPlayer? (True/False) ")
    if gameType == "True":
        whoStarts = input("Does player starts the game?? (True/False) ")
    else:
        whoStarts = "True"

    game = MyGame(size, gameType, whoStarts)
    game.playGame()


if __name__ == "__main__":
    main()
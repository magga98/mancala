# Mancala
This is a simple implementation of Mancala for Computational Intelligence 2023.


## Implementation
The game logic is located in `mancala/game.py`.


## Running the code
An example of how you can run the game is in `sandbox.py`.

To run the code with playback and debug logging:
```
python sandbox.py
```

When training your model you will probably want to disable playback
and debug logging and enable optimizations to speed up the code.
Use the '-00' flag. It will remove all logging code and assertions
as they are now implemented. If you add more logging without enclosing it in 
`if __debug__:` you will also want to change the logging configuration in `mancala/config.py` to `Config.LOG_LEVEL = logging.ERROR` and `Congig.PLAYBACK = False`.

```
python -OO sandbox.py
```


## Board state
The board is represented as a 14-element array.array.
We can visualize the intitial state as:
```
                     Player 1
                12 11 10  9  8  7
            -------------------------
           |     4  4  4  4  4  4    |
        13 |  0                    0 | 6
           |     4  4  4  4  4  4    |
            -------------------------
                 0  1  2  3  4  5
                     Player 0
```
where the numbers inside the box indicate seeds
and the numbers outside the box indicate the indexes
of the array representing the board state.


## Groups
You should create a directory in `mancala/groups` for your code.
At a minimum you need to create the file `manacala/groups/your_group/action.py`
and provide the function `action` inside that file.
The `action` function should have the signature 
```
def action(board: array.array, legal_actions: Tuple[int, ...] -> int
```
that receives the board state and legal actions and returns an int representing the index of the pit that is chosen.
See `mancala/groups/example_group/action.py`.

NOTE: You are not allowed to change the board within your action function.
If you need to change it, copy it:
```
board = copy(board)
```


## Logs
The game provides two logs located in `logs/`.
- `playback.log` contains the game history in ascii drawings as above.
  An action is indicated by `#` in place of the pit index chosen.
  It is useful for debugging and inspecting how your agent plays.
- `debug.log` is a regular log file.
  The logging info in debug.log is also written to standard output.

The logs can be disabled in the configuration file.


## Config
Configuration parameters are set in `mancala/config.py`.
By default, the log level is `DEBUG` and playback is enabled.


## Dependencies
Use Python 3.9 or above.

Package dependencies are located in requirements.txt (and pyproject.toml).
If you add any depenedencies you should add list them in mancala/groups/your_group/requirements.txt

Or if you use *Poetry* (https://python-poetry.org/) add the package to pyproject.toml using
```
poetry add <package> --group your_group
```
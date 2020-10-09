Navigate using a terminal to the project folder and run the following commands:
```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```
This command will attempt learning from 2000 training episodes and then test the resulting AI agent (player) on 10 games.
```
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```
This command will show you what happens during the training process for 10 games.

Use the following command to understand what each of the command line parameters does:
```
python pacman.py --help
```

# how to connect to the ITU cluster

Total setup time: probably like 1.5 hours, mostly cause everything other than running code is fucking slow as shit.

## step 1: be on the ITU network

or connect to the Forticlient VPN, which requires you to set up microsoft MFA properly.

## step 2: ssh into the cluster

Open up a terminal and type the following:

```
ssh alct@hpc.itu.dk
```

Replace `alct` with your ITU username. It will ask if you want to save the fingerprint, type `yes`. It will then ask for a password. This is the password you use for most ITU things like wifi.

You should see a list of all the hardware they have. Keep this terminal open in the background for now.

### step 2.1: set up key authentication

If you don't want to type the password every time, do the following:  
On your own computer, type the following command:

```
ssh-keygen
```

This will generate auth keys. It will ask for the following:

-   Name: call it `hpc` (it will suggest `id_rsa`, don't accept it)
-   Passphrase: keep it empty.

This will generate 2 files: `hpc` and `hpc.pub`. Now create a folder in your home directory called `.ssh`. Your home directory on Windows is `C:\Users\NAME`. On Macos you can probably just type `cd` into the terminal and it will take you there.  
Once you have created the folder, move the two `hpc` files into it. Create a file called `config` and open it in vscode. The file should contain the following:

```
Host            hpc
    Hostname        hpc.itu.dk
    User            alct
    IdentityFile    ~/.ssh/hpc
```

Replace `alct` with your ITU username.  
Now open up the `hpc.pub` file and copy the contents. Go back to your cluster connection that you opened in step 2. Type the following commands:

```
mkdir -p .ssh
cd .ssh
nano authorized_keys
```

This will open up a text editor. Paste the `hpc.pub` text into this editor. Then press `CTRL+X`, then `Y`, then `Enter`.  
Congrants, you have now told the HPC to trust your private key. Open up a new terminal, and type `ssh hpc`. You should be automatically logged in.

## step 3: starting a job

What you've logged into is kind of a "user computer", not actually part of the cluster. It's the layer right before the cluster.  
To use the cluster, you will have to create Job files. These are files that contain a list of commands that will run on the cluster. Here is an example:

```bash
#!/bin/bash
#SBATCH --partition brown     # There are red and brown partitions, we only have access to brown.
#SBATCH --time 02:00:00       # How long the job will run for. You cannot increase this once the job is started.
#SBATCH --job-name test
#SBATCH --output /home/alct/test-%J.log           # Normal command output will go in this file
#SBATCH --error /home/alct/test-error-%J.log      # Commands that give errors will go in this file
#SBATCH --gres=gpu            # Request a GPU. If you don't include this, we only get CPUs

echo "hello world"
```

The only thing this job does it output "hello world" to the test-%J.log file. %J will get replaced with the job ID.  
Note the time parameter: in the example it is set to 2 hours, but the command finish instantly. This will not cause the job to end, so you have to end it manually or the cluster will give you lower priority for keeping idle jobs. See step 5 for the important commands.

### step 3.1: installing packages

We want a job that sets up and installs a bunch of python packages.  
Start by creating a folder for the NLP project:

```
cd ~ # goes home
mkdir work # create work folder
cd work
```

We can then tell the HPC to create a python environment in this folder. Use `nano` again to create a file with `nano install.job`. Note the differences to the above:

```bash
#!/bin/bash
#SBATCH --partition brown
#SBATCH --time 01:00:00
#SBATCH --job-name pythonvenvinstaller
#SBATCH --output /home/alct/work/installer-%J.log
#SBATCH --error /home/alct/work/installer-error-%J.log

echo "Loading python"
module load Python/3.10.4-GCCcore-11.3.0 # gives up access to python 3.10

echo "Creating virtual environment"
# you basically always need to use full paths cause the shell kinda sucks
python -m venv /home/alct/work/venv # create the python environment
source /home/alct/work/venv # activate it

# install lots of shit
echo "Installing packages"
/home/alct/work/venv/bin/pip3 install jupyter notebook ipykernel datasets transformers torch seqeval evaluate accelerate

echo "Done!"
```

REMEMBER TO REPLACE `alct` WITH YOUR USERNAME.  
Now run the job using `sbatch install.job`. Wait a bit and you should see it in the job queue by typing `squeue -u alct` (you username, not alct) along with its id, time left, which computer it is running on etc.  
Once the job gets assigned a machine, it will create the two .log files which you can see with `ls`. Once you see the file, type `tail -f installer-<jobid>.job`. This will show you the output of the job in real time (you can exit with `CTRL+C`). It will take a couple of minutes to begin showing anything, but you should eventually see that it begins installing pip stuff.

The cluster is pretty slow so it can take a while for it to finish installing. Like 30-60 minutes time. (While you're waiting, do step 3.2)  
Once you see the `Done!`, it is finished. Check if the job is still running with the queue command. If it is, cancel it with `scancel <jobid>`.

Now you should also see a `venv` folder if you do `ls`.

### step 3.2: connecting with vscode

Okay this step fucking sucks. The hpc is running such an old server that vscode doesn't really support it, but you NEED to connect with vscode if you want jupyter notebooks to function.

1. Open up vscode.
2. Install the extension called "Remote - SSH" if it isn't already.
3. `CTRL+SHIFT+P` and select "Remote-SSH: Connect to Host..."
4. You should see `hpc` in the list. Select it.
5. VSCode will now tell you that it is downloading and setting up VS Code Server. This will take a while. At some point it might say that the server is too old, ignore it.
6. Wait until it is done loading in the bottom left corner. Don't touch anything until it is done or you will have to start over. Once it is done with this, you might see a small clock icon next to the extensions. That means it is installing extensions on the remote server. Wait until it is done.
7. Once it is done, open the file browser and click "Open Folder". Now you wait again... Eventually you should see a list of folders pop up.
8. You will see it is currently set to `/home/alct`. Scroll down the list and click on the `work` folder you created earlier. Then click "OK".
9. The window will reload, and it will appear to install VS Code Server again (this time will be much quicker) (If it asks to trust the author, click yes). When it is done, open the file browser. Congrats, you should see the two `.log` files, the `.job` file and the `venv` folder.
10. Now that VS Code is connected, you want to install the following extensions: Jupyter, Jupyter Cell Tags, Jupyter Slide Show, Pylance, Python, Python Debugger, Jupyter Notebook Renderers, Jupyter Keymap. Make sure they are installed on the Remote machine too. This is also very slow.

Now VS Code is ready to run Jupyter Notebooks on the HPC. Create a new `notebook.job` file (you can even use vscode now) with the following contents (replace alct with your username as usual):

```bash
#!/bin/bash
#SBATCH --partition brown
#SBATCH --cpus-per-task=1        # Schedule a single core, we don't need anything more
#SBATCH --time 02:00:00
#SBATCH --job-name notebook
#SBATCH --output /home/alct/work/jupyter-notebook-%J.log
#SBATCH --error /home/alct/work/jupyter-notebook-error-%J.log
#SBATCH --gres=gpu               # Schedule a GPU
port=$(shuf -i8000-9999 -n1) # give us a random port

module load Python/3.10.4-GCCcore-11.3.0 # gives up access to python 3.10

# Start the server
source /home/alct/work/venv/bin/activate
/home/alct/work/venv/bin/jupyter-notebook --no-browser --port=${port} --ip=0.0.0.0
```

sbatch this job once the install.job is finished. If it isn't finished yet, move on the step 3.3.

### step 3.3: cloning our code

easy enough, using the command line, make sure you're in the work folder. Then do `git clone https://github.com/ITUprojects/NLP`. Now all of our code should be in `~/work/NLP`.

## step 4: running code

After step 3.2, you should have a running Jupyter Notebook server. Run `tail -f jupyter-notebook-<jobid>.job` in one terminal, and open another terminal where you do `tail -f jupyter-notebook-error-<jobid>.job`. Then open up a jupyter notebook in vscode.  
In the top-right corner, click "Select Kernel", or if there is a python version listed, click it.  
Click "Existing jupyter servers". In one of the two logs you are following above (probably the error one), it should eventually give you a couple links that look something like this:

```log
    To access the server, open this file in a browser:
        file:///home/alct/.local/share/jupyter/runtime/jpserver-32596-open.html
    Or copy and paste one of these URLs:
        http://desktop18:8163/tree?token=4ddf00d94b3597e92f2bfcc852bb5e84bac769cb64c65340
        http://127.0.0.1:8163/tree?token=4ddf00d94b3597e92f2bfcc852bb5e84bac769cb64c65340
```

Copy the one that syas `http://desktop18:...`. Yours might be slightly different, but just take the one that isn't `127.0.0.1`. Paste the link into the VS Code text box that says "Enter a remote URL or select a remote server". Press enter 2 times until it asks you which kernel you want to use. Choose the one that says "ipykernel".

If you end up needing any new pip packages, you are going to have to install them using them using the `install.job` job. Just copy the pip install line, put a hashtag (#) in front of one of them and remove every package except the one you want to install. This should make sure it doesn't try to reinstall every single package again.

You can now run code on the GPU. Check GPU availability with the following code:

```py
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

```log
True
NVIDIA GeForce GTX 1080 Ti
```

(the first time importing something will always be a little slow). As you can see, it's using a 1080 Ti. This GPU is pretty weak for training large models, so once it's time to actually crunch the numbers, see step 5.1 for how to specify a bigger one.

## step 5: commands and slurm jobs

list of the important commands when running jobs on the cluster.

| Command           | Description                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `sbatch file.job` | Dispatches a job to the cluster, scheduling it for a specific computer based on the requirements at the top of the file.          |
| `squeue -u alct`  | See a list of jobs currently dispatched by user `alct`.                                                                           |
| `scancel <jobid>` | Cancel a job based on its ID. Always do this when you are done with a specific job to make sure your account is in good standing. |
| `sacct -u alct`   | Get status of the jobs dispatched by `alct`. This is kinda like `squeue`, but shows different information.                        |

### step 5.1: sheduling more powerful hardware

By default, having `gres=gpu` means the system will randomly assign you a GPU. Most of the GPUs are shit though, so you might want to specify a specific type. When you log in to the cluster, they show you the list of GPUs they have. You can change your job to something like this to get one of the good ones:

```bash
#SBATCH --gres=gpu:a30:1
```

This basically says that you will only accept A30 GPUs, and you only need 1 of them. Check out [hpc.itu.dk/scheduling/templates/gpu/](http://hpc.itu.dk/scheduling/templates/gpu/) for a list of GPUs you can request.
Note that there are much fewer A30 GPUs that 1080 Tis. This means it might take longer for you job to start. Use the queue command to see if it has started yet. If it hasen't the remaining time will still be full, and it should says something in the "reason" field that explains what it is waiting for.

## step 6: long-running jobs

You probably don't want to keep vscode on the whole time in a jupyter notebook, so you can just create a new job that, instead of starting a jupyter notebook server, just runs a `.py` file. Just include this line instead of the jupyter server one:

```bash
/home/alct/work/venv/bin/python3 /home/alct/work/NLP/train_models_or_some_shit.py
```

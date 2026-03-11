***********
Whisper.cpp
***********

Whisper.cpp is an automatic speech recognition tool that converts spoken audio into text. It is well suited to processing large collections 
of audio recordings, such as interviews, lectures, oral histories, podcasts, and produces transcripts in several formats at once:

- **Plain text** (``.txt``): one file per recording, ready to read or search
- **SRT subtitles** (``.srt``): compatible with most video players and editing tools
- **WebVTT subtitles** (``.vtt``): used by web-based video players
- **Detailed JSON** (``.json``): includes word-level timestamps and confidence scores

Running it on the BEDE HPC cluster lets you process dozens or hundreds of files in parallel overnight, rather than transcribing them one at a 
time on a laptop.

The software is available as a module on BEDE and requires no installation on your part. The command line tools used in this tutorial should be
available in Microsoft PowerShell as well as Linux and MacOS shells.


Walkthrough
-----------

This tutorial walks you through transcribing a folder of audio files on the BEDE HPC cluster using the provided scripts. It assumes you already 
have a BEDE account and are comfortable typing commands into a terminal. No programming experience is required.

You will need two terminal windows open: one connected to BEDE (for running jobs), and one on your own computer (for transferring files). 
Commands that must be typed on your **local computer** are shown with a different background colour; all other commands are run **on BEDE**.

Here is an overview of the steps:

1. Upload your audio files to BEDE, or download them there directly
2. Create a short job script that tells BEDE what to transcribe and where to save the results
3. Submit the job and wait for the transcripts to appear

To connect to BEDE, open a terminal on your local computer and run (as described in the "Using Bede" section):

.. code-block:: bash
   :class: local

   ssh <bede-username>@bede.dur.ac.uk

Audio files can be large, so you should store them in the ``/nobackup`` area rather than your home directory. The ``/nobackup`` 
area has much more space and is the right place for large datasets. Note that files there are not backed up, so keep originals elsewhere. 
In the shell you used to log in, navigate there (replace ``<project-id>`` with your project code):

.. code-block:: bash

   cd /nobackup/projects/<project-id>

   ## create a user folder if you do not already have one
   mkdir -p $USER


There are two ways to get audio files onto BEDE: uploading from your own computer, or downloading directly from the internet on the cluster. 
The tutorial assumes you keep the BEDE-connected terminal open throughout.

Transcribing local files
=========================

If your audio files are on your own computer, you transfer them using ``scp`` (Secure Copy: a standard command-line tool
for sending files over a network connection). Open a **new** terminal on your local computer. Do not use the one already connected to BEDE.

The commands differ slightly depending on your operating system:

.. tabs::

   .. group-tab:: Linux / macOS

      Navigate into the parent folder of your audio files:

      .. code-block:: bash
         :class: local

         cd /path/to/your/<audio-folder>

      Move up one level so you can copy the whole folder across to BEDE. The ``-r`` option means "recursive": it copies the folder and everything inside it.

      .. code-block:: bash
         :class: local
        
         cd .. #go to previous folder to copy the audio folder with its contents.
         scp -r <audio-folder> <bede-username>@bede.dur.ac.uk:/nobackup/projects/<project>/<bede-username>

   .. group-tab:: Windows PowerShell

      On Windows, your files may be on a different drive (e.g. ``D:``). Switch to that drive first,
      then navigate into the parent folder of your audio files:

      .. code-block:: powershell
         :class: local

         D:
         cd D:\path\to\your\<audio-folder>

      Move up one level so you can copy the whole folder across to BEDE. The ``-r`` option means "recursive": it copies the folder and everything inside it.

      .. code-block:: powershell
         :class: local

         cd .. #go to previous folder to copy the audio folder with its contents.
         scp -r <audio-folder> <bede-username>@bede.dur.ac.uk:/nobackup/projects/<project>/<bede-username>

      .. note::

         ``scp`` is available in PowerShell on Windows 10 and later. If your version of Windows does
         not have it, you can install `OpenSSH <https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh>`_
         or use a tool such as `WinSCP <https://winscp.net>`_ which provides a graphical interface.

Transcribing files from the internet
====================================

Files from the internet can be downloaded directly on the cluster using wget or curl.

First we create a folder in our user folder and then download the file on the cluster, using the console connected to the cluster.
The example below downloads a Nobel lecture by Ada E. Yonath. Replace the URL with the address of your own file.

.. code-block:: bash

   cd /nobackup/projects/<project-id>/<bede-username>
   mkdir -p audio_files
   cd audio_files
   wget https://nobel-videocdn01.azureedge.net/video/lecture_2009_che_yonath-intro_01_496.mp4

Either way, your audio files are now on BEDE and ready to be transcribed.

The next step is to create a **job script** — a short text file that tells BEDE's job scheduler (called SLURM) what program to run, on which files,
and for how long. SLURM queues your request, finds a free GPU node, runs it, and writes the output to a folder you choose. You do not
need to stay logged in while it runs.

Choose the right script depending on how many files you have:

- **Fewer than ~50 files (up to ~10 hours of audio total):** use the single-job script, which processes the whole folder in one go.
- **More than ~50 files:** use the job array script, which splits the work across several parallel jobs and is much faster.

Creating a job script
======================

Create a new text file on BEDE using the ``nano`` editor (a simple terminal text editor; use the arrow keys to move, ``Ctrl+O`` to save, 
``Ctrl+X`` to exit):

.. code-block:: bash

   nano transcribe.sh

Copy the appropriate script below into the editor. You only need to edit the lines in the ``USER CONFIGURATION`` section: replace 
``<project>`` with your project code (e.g. ``bddur99``) and adjust the paths to your audio folder and desired output folder. 
The ``ENV_FILE`` line is optional and is explained in the next section.

BEDE currently has two different sets of computers called nodes available. Which of the two you use can be chosen with the ``#SBATCH --partition=gpu``
line. The current setting (``gpu``) chooses nodes using V100 GPUs, which are available in higher numbers. 
For running the job on a GraceHopper node, you can change the partition to ``gh`` (TODO: Support this).
These nodes are more powerful than needed for most transcription jobs, and as they are less widely available, 
jobs often wait longer in the queue before starting.

.. tabs::

   .. group-tab:: Small number of files

      If you have a small number of files to transcribe, you can use the following slurm script. It will process all audio files in the 
      specified folder in one job.

      .. code-block:: bash

         #!/bin/bash
         ################################################################################
         # Whisper.cpp Folder Transcription — Single-node (no job array)
         # Processes all audio files in a folder in one job.
         ################################################################################

         # ── SLURM directives ──────────────────────────────────────────────────────────
         #SBATCH --account=<project>   # Run job under project <project>
         #SBATCH --time=3:00:0
         #SBATCH --job-name=whisper_folder

         # GPU node
         #SBATCH --partition=gpu
         #SBATCH --nodes=1
         #SBATCH --gres=gpu:1

         # Log files (%j = job ID)
         #SBATCH --output=logs/%j/folder.out
         #SBATCH --error=logs/%j/folder.err

         # ── USER CONFIGURATION ────────────────────────────────────────────────────────
         PROJECT=<project> # Your BEDE project name (e.g. myproject)

         # Folder containing your audio files
         AUDIO_DIR=/nobackup/projects/$PROJECT/$USER/audio_files

         # Audio file extension to process (e.g. mp3, wav, flac, m4a)
         AUDIO_EXT=mp3

         # Where transcription output will be written (txt/, vtt/, srt/, json/ auto-created)
         OUTDIR=/nobackup/projects/$PROJECT/$USER/whisper_output

         # Whisper parameters (see scripts/config/whisper.env to adjust quality/language/etc.)
         ENV_FILE="/path/to/my.env"

         # ── END USER CONFIGURATION ────────────────────────────────────────────────────

         # Load whisper.cpp module

         # TODO Replace this with the final module load command.
         export MODULEPATH=$MODULEPATH:/nobackup/projects/bddur53/pnruth/modules/ppc64le
         module load whisper.cpp
         # ENDTODO

         # Load whisper parameters from config file
         if [ -f "$ENV_FILE" ]; then
            set -a
            source <(sed 's/#.*//' "$ENV_FILE" | grep -v '^\s*$')
            set +a
            echo "Loaded config from: $ENV_FILE"
         else
            echo "WARNING: Config file not found: $ENV_FILE (worker will use built-in defaults)"
         fi

         # Export variables needed by the worker
         export AUDIO_DIR OUTDIR AUDIO_EXT MODEL_DIR
         export EXTRA_ARGS="${EXTRA_ARGS:-}"

         # Hand off all processing logic to the worker script
         bash "$WORKER_PATH"

   .. group-tab:: Large number of files

      For a large number of files, it is more efficient to use a job array. The following slurm script will process each audio file 
      in the specified folder as a separate job in a job array. This allows for better parallelization and resource management.

      You also need to set ``#SBATCH --array=1-N``, where ``N`` is the number of parallel jobs to run. A good rule of thumb is one job per 20 files. To count your files, run:

      .. code-block:: bash

         find /nobackup/projects/<project>/<bede-username>/audio_files -name "*.mp3" | wc -l

      Then divide that number by 20 and round up; that is your ``N``. For example, 75 files ÷ 20 = 3.75, so set ``--array=1-4``. Each job automatically figures out which slice of files it is responsible for, so you do not need to split them manually.

      .. code-block:: bash

         #!/bin/bash
         ################################################################################
         # Whisper.cpp Batch Transcription — SLURM Submit Script
         ################################################################################

         # ── SLURM directives ──────────────────────────────────────────────────────────
         #SBATCH --account=<project>
         #SBATCH --time=2:00:0
         #SBATCH --job-name=whisper_transcribe

         #SBATCH --array=1-4

         # GPU node
         #SBATCH --partition=gpu
         #SBATCH --nodes=1
         #SBATCH --gres=gpu:1

         # Log files (%A = job ID, %a = array task ID)
         #SBATCH --output=logs/%A/task_%a.out
         #SBATCH --error=logs/%A/task_%a.err

         # ── USER CONFIGURATION ────────────────────────────────────────────────────────
         PROJECT=<project> # Your BEDE project name (e.g. myproject)

         # Folder containing your audio files
         AUDIO_DIR=/nobackup/projects/$PROJECT/$USER/audio_files

         # Audio file extension to process (e.g. mp3, wav, flac, m4a)
         AUDIO_EXT=mp3

         # Where transcription output will be written (txt/, vtt/, srt/, json/ auto-created)
         OUTDIR=/nobackup/projects/$PROJECT/$USER/whisper_output

         # Whisper parameters (see scripts/config/whisper.env to adjust quality/language/etc.)
         ENV_FILE=/path/to/my.env

         # ── END USER CONFIGURATION ────────────────────────────────────────────────────

         # TODO Replace this with the final module load command.
         export MODULEPATH=$MODULEPATH:/nobackup/projects/bddur53/pnruth/modules/ppc64le
         module load whisper.cpp
         # ENDTODO

         # Load whisper parameters from config file
         if [ -f "$ENV_FILE" ]; then
            set -a
            source <(sed 's/#.*//' "$ENV_FILE" | grep -v '^\s*$')
            set +a
            echo "Loaded config from: $ENV_FILE"
         else
            echo "WARNING: Config file not found: $ENV_FILE (worker will use built-in defaults)"
         fi

         # Export variables needed by the worker
         export AUDIO_DIR OUTDIR AUDIO_EXT MODEL_DIR
         export EXTRA_ARGS="${EXTRA_ARGS:-}"

         # Hand off all processing logic to the worker script
         bash "$WORKER_PATH"

Save the file and submit it to the queue:

.. code-block:: bash

   sbatch transcribe.sh

BEDE will print a job ID, for example ``Submitted batch job 123456``. You can check whether your job is running or waiting in the queue with:

.. code-block:: bash

   squeue --me

When the job finishes, the transcripts will be in the output folder you set in ``OUTDIR``, organised into subfolders:

- ``OUTDIR/txt/`` — plain text transcripts
- ``OUTDIR/srt/`` — SRT subtitle files
- ``OUTDIR/vtt/`` — WebVTT subtitle files
- ``OUTDIR/json/`` — detailed JSON with timestamps

If something went wrong, check the log file in ``logs/<job-id>/`` for error messages.

Getting the files back to your computer
=======================================

Once your job has finished, copy the output folder from BEDE to your local computer using ``scp``.
In a terminal **on your local computer** (not the one connected to BEDE), go to the folder where you want to save the transcripts using ``cd``, then run:

.. code-block:: bash
   :class: local

   scp -r <bede-username>@bede.dur.ac.uk:/nobackup/projects/<project>/<bede-username>/whisper_output ./whisper_output

This will download the entire ``whisper_output`` folder into your current directory. Inside it you will find four subfolders:
``txt/``, ``srt/``, ``vtt/``, and ``json/``, each containing one file per recording.

If you only want the plain text transcripts and not the subtitle or JSON files, you can download just that subfolder:

.. code-block:: bash
   :class: local

   scp -r <bede-username>@bede.dur.ac.uk:/nobackup/projects/<project>/<bede-username>/whisper_output/txt ./transcripts


This should enable you to produce transcripts. If you encounter any issues or want to customise the transcription process, 
read on for more details about the configuration options.


Transpiling with FFmpeg
=======================

To be added



More customisation with an ``env`` file
=======================================

The ``ENV_FILE`` is optional. The built-in defaults work well for most English-language recordings,
so you can skip this section entirely on your first run.

If you do need to change something — most commonly the language — create a plain text file containing
only the lines you want to override. Save it anywhere on BEDE and set ``ENV_FILE`` in your job script
to point at it. Everything you leave out keeps its default value.

Common configurations
---------------------

**Non-English audio**

.. code-block:: bash

  WHISPER_LANGUAGE=de   # replace 'de' with your language code

Common codes: ``de`` German, ``fr`` French, ``es`` Spanish, ``nl`` Dutch, ``it`` Italian,
``pt`` Portuguese, ``ja`` Japanese, ``zh`` Chinese. Use ``auto`` to let Whisper detect
the language automatically (slightly slower).

**Translate speech into English** (e.g. a French interview delivered as an English transcript)

.. code-block:: bash

  WHISPER_LANGUAGE=fr
  TRANSLATE=true

**Faster processing, slightly lower accuracy** (useful for a quick first pass)

.. code-block:: bash

  BEST_OF=2
  BEAM_SIZE=2

**Plain text output only** (skip subtitle and JSON files to save space)

.. code-block:: bash

  OUTPUT_FORMATS=-otxt

Pointing your job script at the env file
-----------------------------------------

Save the file on BEDE, for example as ``my_project.env`` alongside your job script, then set
``ENV_FILE`` in the ``USER CONFIGURATION`` section of your script:

.. code-block:: bash

   ENV_FILE=/nobackup/projects/<project>/<bede-username>/my_project.env

Alternatively, pass it at submission time without editing the script at all:

.. code-block:: bash

   sbatch --export=ALL,ENV_FILE=/nobackup/projects/<project>/<bede-username>/my_project.env transcribe.sh

Full settings reference
-----------------------

The tables below list every available setting. You will not need most of these for typical transcription work.

.. rubric:: Language

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Description
   * - ``WHISPER_LANGUAGE``
     - ``en``
     - Spoken language of your audio. Use ``auto`` for automatic detection (slightly slower).
       Common codes: ``en``, ``de``, ``fr``, ``es``, ``zh``, ``ja``.
   * - ``TRANSLATE``
     - ``false``
     - Set to ``true`` to translate the audio into English instead of transcribing it.
       Only meaningful for non-English source audio.

.. rubric:: Output formatting

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Description
   * - ``OUTPUT_FORMATS``
     - ``-otxt -ovtt -osrt -ojf``
     - Space-separated output format flags. Remove any you do not need.
       ``-otxt`` plain text, ``-ovtt`` WebVTT subtitles, ``-osrt`` SRT subtitles,
       ``-ojf`` JSON with full token-level timestamps and probabilities.
   * - ``MAX_LEN``
     - ``100``
     - Maximum characters per transcript segment. ``0`` produces one large block.
       ``100`` gives sentence-friendly chunks suitable for subtitles.
       The whisper-cli default is ``0``.
   * - ``MAX_CONTEXT``
     - ``32``
     - Number of previous tokens Whisper remembers across segments. ``-1`` keeps all
       context; ``0`` decodes each segment independently. The whisper-cli default is ``-1``.
   * - ``PROMPT``
     - *see below*
     - Initial text prompt to steer Whisper's style and vocabulary. Helpful for
       punctuation, capitalisation, and domain-specific terms.
       Default: ``Transcribe with proper punctuation including capitalisation and full stops.``

.. rubric:: Accuracy and speed

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Description
   * - ``BEST_OF``
     - ``8``
     - Number of candidate transcriptions generated per chunk (best-of-N sampling).
       Higher values improve accuracy at the cost of speed. The whisper-cli default is ``5``.
   * - ``BEAM_SIZE``
     - ``8``
     - Beam search width — how many paths are explored simultaneously.
       Higher values improve accuracy at the cost of speed. The whisper-cli default is ``5``.
   * - ``TEMPERATURE``
     - ``0.5``
     - Sampling temperature (0.0 = fully deterministic). Higher values introduce more
       variation. The whisper-cli default is ``0.00``.
   * - ``TEMPERATURE_INC``
     - ``0.3``
     - Temperature increment applied on each fallback retry (only relevant when
       ``NO_FALLBACK=false``). The whisper-cli default is ``0.20``.
   * - ``NO_FALLBACK``
     - ``true``
     - When ``true``, Whisper will not retry uncertain segments with a higher temperature.
       Keeps results deterministic; set to ``false`` for more aggressive recovery.

.. rubric:: Voice Activity Detection (VAD)

VAD filters silence and background noise *before* passing audio to Whisper, preventing hallucinated
text during quiet periods. It is strongly recommended for recordings with background noise or music.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Variable
     - Default
     - Description
   * - ``VAD_MODEL``
     - ``ggml-silero-v5.1.2.bin``
     - Silero VAD model file. Must be present in the module's model directory.
   * - ``VAD_THRESHOLD``
     - ``0.1``
     - Speech confidence threshold (0.0–1.0). Lower values retain more quiet speech but
       risk passing through background noise. The whisper-cli default is ``0.50``.
   * - ``VAD_MIN_SPEECH_MS``
     - ``300``
     - Minimum duration (ms) a speech burst must last to be kept. Shorter bursts
       (coughs, clicks) are discarded. The whisper-cli default is ``250``.
   * - ``VAD_MIN_SILENCE_MS``
     - ``200``
     - Minimum silence gap (ms) between speech segments before they are split into
       separate chunks. The whisper-cli default is ``100``.

.. rubric:: Models

Whisper is available in several sizes trading off speed against accuracy. The scripts automatically
try the highest-quality model first and fall back to the smaller one if the GPU runs out of memory.

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Variable
     - Default
     - Description
   * - ``MODEL_PRIMARY``
     - ``ggml-large-v3.bin``
     - Highest-accuracy model (~3.9 GB VRAM). Used first.
   * - ``MODEL_FALLBACK``
     - ``ggml-medium.bin``
     - Lighter model (~2.1 GB VRAM). Used when the primary model runs out of memory.

.. rubric:: Advanced: quality thresholds

These settings control when the script considers a transcribed segment unreliable and
falls back to the smaller model. The defaults are well-tuned and rarely need changing.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Description
   * - ``ENTROPY_THOLD``
     - ``2.8``
     - Maximum token entropy allowed before a segment is marked as failed. Higher values
       tolerate more uncertainty. The whisper-cli default is ``2.40``.
   * - ``LOGPROB_THOLD``
     - ``-1.0``
     - Minimum average log-probability for a segment. ``-1.0`` disables this check.
       The whisper-cli default is ``-1.00``.
   * - ``NO_SPEECH_THOLD``
     - ``0.3``
     - Probability above which a segment is classified as silence and discarded.
       Lower values keep more borderline speech. The whisper-cli default is ``0.60``.
   * - ``WORD_THOLD``
     - ``0.05``
     - Minimum per-word timestamp confidence. The whisper-cli default is ``0.01``.




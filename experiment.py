#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Wed Mar 27 18:10:23 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
# variables that are constant for the experiment
import random
from numpy.random import choice

fixation_size = (15, 15)
circle_size = (70,70)
rectangle_size = (70,70)
color = "green"
colors = ["red", "green", "blue"]
image_size = (200, 200)
possible_positions = [(0,250), (0,-250), (250, 0), (-250, 0)]
conditions = ['present_emotional', 'present_neutral', 'absent_emotional', 'absent_neutral']
weights = [1, 1, 1, 1]
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'experiment'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath="/Users/pushkarsingh/Documents/01 University/03 Self Prioritisation Research/11 Ananya's Experiment/experiment/experiment.py",
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=False, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "load_variables" ---
    # Run 'Begin Experiment' code from code
    distractor_color = ""
    target_color = ""
    correct_response = ""
    current_condition = ""
    screen_components = []
    circle = ""
    action_target = ""
    action_distractor = ""
    neutral_red_square = ""
    
    
    # --- Initialize components for Routine "experiment_introduction" ---
    experiment_intro_text = visual.TextStim(win=win, name='experiment_intro_text',
        text="Welcome to the experiment!\n\npress 'spacebar' to continue.",
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcome_response = keyboard.Keyboard()
    
    # --- Initialize components for Routine "action_introduction" ---
    action_text = visual.TextStim(win=win, name='action_text',
        text="In this phase, you will be shown three circles, one square and an emotional / non emotional image in the center. Your tasks is to ignore the emotional / non emotional image and click on the sqaure as fast as you can.\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    action_intro_respone = keyboard.Keyboard()
    
    # --- Initialize components for Routine "action_fixation_routine" ---
    action_fixation = visual.ShapeStim(
        win=win, name='action_fixation', vertices='cross',units='pix', 
        size=fixation_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "action" ---
    # Run 'Begin Experiment' code from action_code
    action_circle3 = visual.ShapeStim(
            win=win, name='action_circle3',units='pix', 
            size=circle_size, vertices='circle',
            ori=0.0, pos=[0,0], anchor='center',
            lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
            opacity=None, depth=-3.0, interpolate=True)
    action_response = event.Mouse(win=win)
    x, y = [None, None]
    action_response.mouseClock = core.Clock()
    action_distractor = visual.ImageStim(
        win=win,
        name='action_distractor', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "perception_introduction" ---
    perception_text = visual.TextStim(win=win, name='perception_text',
        text="In this phase, you will be shown three circles, one square and an emotional / non emotional image in the center. Now the circle and the target (square)  will keep changing color between red, green and blue. Your task is to ignore the emotional / non emotional image and report the color of the target using the following keys :\n\nRed - 'r'\nGreen - 'g'\nBlue - 'b'\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    perception_intro_respone = keyboard.Keyboard()
    
    # --- Initialize components for Routine "perception" ---
    perception_fixation = visual.ShapeStim(
        win=win, name='perception_fixation', vertices='cross',units='pix', 
        size=fixation_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    perception_circle1 = visual.ShapeStim(
        win=win, name='perception_circle1',units='pix', 
        size=circle_size, vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    perception_circle2 = visual.ShapeStim(
        win=win, name='perception_circle2',units='pix', 
        size=circle_size, vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    perception_circle3 = visual.ShapeStim(
        win=win, name='perception_circle3',units='pix', 
        size=circle_size, vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    perception_target = visual.Rect(
        win=win, name='perception_target',units='pix', 
        width=rectangle_size[0], height=rectangle_size[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    perception_distractor = visual.ImageStim(
        win=win,
        name='perception_distractor', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    perception_response = keyboard.Keyboard()
    
    # --- Initialize components for Routine "thank_you" ---
    thank_you_text = visual.TextStim(win=win, name='thank_you_text',
        text='thank you for your participation. :))',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "load_variables" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('load_variables.started', globalClock.getTime())
    # keep track of which components have finished
    load_variablesComponents = []
    for thisComponent in load_variablesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "load_variables" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in load_variablesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "load_variables" ---
    for thisComponent in load_variablesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('load_variables.stopped', globalClock.getTime())
    # the Routine "load_variables" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "experiment_introduction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('experiment_introduction.started', globalClock.getTime())
    welcome_response.keys = []
    welcome_response.rt = []
    _welcome_response_allKeys = []
    # keep track of which components have finished
    experiment_introductionComponents = [experiment_intro_text, welcome_response]
    for thisComponent in experiment_introductionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "experiment_introduction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *experiment_intro_text* updates
        
        # if experiment_intro_text is starting this frame...
        if experiment_intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_intro_text.frameNStart = frameN  # exact frame index
            experiment_intro_text.tStart = t  # local t and not account for scr refresh
            experiment_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_intro_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            experiment_intro_text.status = STARTED
            experiment_intro_text.setAutoDraw(True)
        
        # if experiment_intro_text is active this frame...
        if experiment_intro_text.status == STARTED:
            # update params
            pass
        
        # *welcome_response* updates
        waitOnFlip = False
        
        # if welcome_response is starting this frame...
        if welcome_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_response.frameNStart = frameN  # exact frame index
            welcome_response.tStart = t  # local t and not account for scr refresh
            welcome_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_response, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_response.started')
            # update status
            welcome_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_response.status == STARTED and not waitOnFlip:
            theseKeys = welcome_response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_response_allKeys.extend(theseKeys)
            if len(_welcome_response_allKeys):
                welcome_response.keys = _welcome_response_allKeys[0].name  # just the first key pressed
                welcome_response.rt = _welcome_response_allKeys[0].rt
                welcome_response.duration = _welcome_response_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in experiment_introductionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "experiment_introduction" ---
    for thisComponent in experiment_introductionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('experiment_introduction.stopped', globalClock.getTime())
    # check responses
    if welcome_response.keys in ['', [], None]:  # No response was made
        welcome_response.keys = None
    thisExp.addData('welcome_response.keys',welcome_response.keys)
    if welcome_response.keys != None:  # we had a response
        thisExp.addData('welcome_response.rt', welcome_response.rt)
        thisExp.addData('welcome_response.duration', welcome_response.duration)
    thisExp.nextEntry()
    # the Routine "experiment_introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "action_introduction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('action_introduction.started', globalClock.getTime())
    action_intro_respone.keys = []
    action_intro_respone.rt = []
    _action_intro_respone_allKeys = []
    # keep track of which components have finished
    action_introductionComponents = [action_text, action_intro_respone]
    for thisComponent in action_introductionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "action_introduction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *action_text* updates
        
        # if action_text is starting this frame...
        if action_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            action_text.frameNStart = frameN  # exact frame index
            action_text.tStart = t  # local t and not account for scr refresh
            action_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(action_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            action_text.status = STARTED
            action_text.setAutoDraw(True)
        
        # if action_text is active this frame...
        if action_text.status == STARTED:
            # update params
            pass
        
        # *action_intro_respone* updates
        waitOnFlip = False
        
        # if action_intro_respone is starting this frame...
        if action_intro_respone.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            action_intro_respone.frameNStart = frameN  # exact frame index
            action_intro_respone.tStart = t  # local t and not account for scr refresh
            action_intro_respone.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(action_intro_respone, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'action_intro_respone.started')
            # update status
            action_intro_respone.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(action_intro_respone.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(action_intro_respone.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if action_intro_respone.status == STARTED and not waitOnFlip:
            theseKeys = action_intro_respone.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _action_intro_respone_allKeys.extend(theseKeys)
            if len(_action_intro_respone_allKeys):
                action_intro_respone.keys = _action_intro_respone_allKeys[0].name  # just the first key pressed
                action_intro_respone.rt = _action_intro_respone_allKeys[0].rt
                action_intro_respone.duration = _action_intro_respone_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in action_introductionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "action_introduction" ---
    for thisComponent in action_introductionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('action_introduction.stopped', globalClock.getTime())
    # check responses
    if action_intro_respone.keys in ['', [], None]:  # No response was made
        action_intro_respone.keys = None
    thisExp.addData('action_intro_respone.keys',action_intro_respone.keys)
    if action_intro_respone.keys != None:  # we had a response
        thisExp.addData('action_intro_respone.rt', action_intro_respone.rt)
        thisExp.addData('action_intro_respone.duration', action_intro_respone.duration)
    thisExp.nextEntry()
    # the Routine "action_introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    action_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('distractors.csv', selection=choice(268, size = 10, replace = False)),
        seed=None, name='action_loop')
    thisExp.addLoop(action_loop)  # add the loop to the experiment
    thisAction_loop = action_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisAction_loop.rgb)
    if thisAction_loop != None:
        for paramName in thisAction_loop:
            globals()[paramName] = thisAction_loop[paramName]
    
    for thisAction_loop in action_loop:
        currentLoop = action_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisAction_loop.rgb)
        if thisAction_loop != None:
            for paramName in thisAction_loop:
                globals()[paramName] = thisAction_loop[paramName]
        
        # --- Prepare to start Routine "action_fixation_routine" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('action_fixation_routine.started', globalClock.getTime())
        # keep track of which components have finished
        action_fixation_routineComponents = [action_fixation]
        for thisComponent in action_fixation_routineComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "action_fixation_routine" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *action_fixation* updates
            
            # if action_fixation is starting this frame...
            if action_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                action_fixation.frameNStart = frameN  # exact frame index
                action_fixation.tStart = t  # local t and not account for scr refresh
                action_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(action_fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                action_fixation.status = STARTED
                action_fixation.setAutoDraw(True)
            
            # if action_fixation is active this frame...
            if action_fixation.status == STARTED:
                # update params
                pass
            
            # if action_fixation is stopping this frame...
            if action_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > action_fixation.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    action_fixation.tStop = t  # not accounting for scr refresh
                    action_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    action_fixation.status = FINISHED
                    action_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in action_fixation_routineComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "action_fixation_routine" ---
        for thisComponent in action_fixation_routineComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('action_fixation_routine.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "action" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('action.started', globalClock.getTime())
        # Run 'Begin Routine' code from action_code
        random.shuffle(possible_positions)
        pos_action_circle1, pos_action_circle2, pos_action_circle3, pos_action_target = possible_positions
        
        random.shuffle(conditions)
        current_condition = random.choices(conditions, weights=weights)[0]
        
        # componenets that make the
        if current_condition == "present_emotional":
            thisExp.addData("Target", "present")
            thisExp.addData("Phase", "action")
            thisExp.addData("Valence", code)
            for i in range(3):
                circle = visual.ShapeStim(
                win=win, name=f'action_circle_{i}',units='pix', 
                size=circle_size, vertices='circle',
                ori=0.0, pos=possible_positions[i], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-3.0, interpolate=True)
                screen_components.append(circle)
            action_distractor = visual.ImageStim(
                win=win,
                name='action_distractor', units='pix', 
                image='default.png', mask=None, anchor='center',
                ori=0.0, pos=(0, 0), size=image_size,
                color=[1,1,1], colorSpace='rgb', opacity=None,
                flipHoriz=False, flipVert=False,
                texRes=128.0, interpolate=True, depth=-6.0)
            screen_components.append(action_distractor)
            action_target = visual.Rect(
                win=win, name='action_target',units='pix', 
                width=rectangle_size[0], height=rectangle_size[1],
                ori=0.0, pos=possible_positions[3], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-4.0, interpolate=True)
            screen_components.append(action_target)   
        elif current_condition == "present_neutral":
            thisExp.addData("Target", "present")
            thisExp.addData("Phase", "action")
            thisExp.addData("Valence", "neutral")
            for i in range(3):
                circle = visual.ShapeStim(
                win=win, name=f'action_circle_{i}',units='pix', 
                size=circle_size, vertices='circle',
                ori=0.0, pos=possible_positions[i], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-3.0, interpolate=True)
                screen_components.append(circle)
            action_target = visual.Rect(
                win=win, name='action_target',units='pix', 
                width=rectangle_size[0], height=rectangle_size[1],
                ori=0.0, pos=possible_positions[3], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-4.0, interpolate=True)
            screen_components.append(action_target)
            neutral_red_square = visual.Rect(
                win=win, name='red_square', units='pix',
                width=image_size[0], height=image_size[1],
                ori=0.0, pos=(0, 0), anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
                opacity=None, depth=-8.0, interpolate=True)
            screen_components.append(neutral_red_square)
        elif current_condition == "absent_emotional":
            thisExp.addData("Target", "Absent")
            thisExp.addData("Phase", "action")
            thisExp.addData("Valence", code)
            for i in range(4):
                circle = visual.ShapeStim(
                win=win, name=f'action_circle_{i}',units='pix', 
                size=circle_size, vertices='circle',
                ori=0.0, pos=possible_positions[i], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-3.0, interpolate=True)
                screen_components.append(circle)
            action_distractor = visual.ImageStim(
                win=win,
                name='action_distractor', units='pix', 
                image='default.png', mask=None, anchor='center',
                ori=0.0, pos=(0, 0), size=image_size,
                color=[1,1,1], colorSpace='rgb', opacity=None,
                flipHoriz=False, flipVert=False,
                texRes=128.0, interpolate=True, depth=-6.0)
            screen_components.append(action_distractor)
        elif current_condition == 'absent_neutral':
            thisExp.addData("Target", "absent")
            thisExp.addData("Phase", "action")
            thisExp.addData("Valence", "neutral")
            for i in range(4):
                circle = visual.ShapeStim(
                win=win, name=f'action_circle_{i}',units='pix', 
                size=circle_size, vertices='circle',
                ori=0.0, pos=possible_positions[i], anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
                opacity=None, depth=-3.0, interpolate=True)
                screen_components.append(circle)
            neutral_red_square = visual.Rect(
                win=win, name='red_square', units='pix',
                width=image_size[0], height=image_size[1],
                ori=0.0, pos=(0, 0), anchor='center',
                lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
                opacity=None, depth=-8.0, interpolate=True)
            screen_components.append(neutral_red_square)
        else:
            print("something went wrong, option is not available")
            
        for component in screen_components:
            component.setAutoDraw(True)
        # setup some python lists for storing info about the action_response
        action_response.x = []
        action_response.y = []
        action_response.leftButton = []
        action_response.midButton = []
        action_response.rightButton = []
        action_response.time = []
        action_response.clicked_name = []
        gotValidClick = False  # until a click is received
        action_distractor.setImage(image)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        actionComponents = [action_response, action_distractor, key_resp]
        for thisComponent in actionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "action" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *action_response* updates
            
            # if action_response is starting this frame...
            if action_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                action_response.frameNStart = frameN  # exact frame index
                action_response.tStart = t  # local t and not account for scr refresh
                action_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(action_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('action_response.started', t)
                # update status
                action_response.status = STARTED
                action_response.mouseClock.reset()
                prevButtonState = action_response.getPressed()  # if button is down already this ISN'T a new click
            
            # if action_response is stopping this frame...
            if action_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > action_response.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    action_response.tStop = t  # not accounting for scr refresh
                    action_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('action_response.stopped', t)
                    # update status
                    action_response.status = FINISHED
            if action_response.status == STARTED:  # only update if started and not finished!
                buttons = action_response.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([action_target, neutral_red_square], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(action_response):
                                gotValidClick = True
                                action_response.clicked_name.append(obj.name)
                        x, y = action_response.getPos()
                        action_response.x.append(x)
                        action_response.y.append(y)
                        buttons = action_response.getPressed()
                        action_response.leftButton.append(buttons[0])
                        action_response.midButton.append(buttons[1])
                        action_response.rightButton.append(buttons[2])
                        action_response.time.append(action_response.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *action_distractor* updates
            
            # if action_distractor is starting this frame...
            if action_distractor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                action_distractor.frameNStart = frameN  # exact frame index
                action_distractor.tStart = t  # local t and not account for scr refresh
                action_distractor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(action_distractor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'action_distractor.started')
                # update status
                action_distractor.status = STARTED
                action_distractor.setAutoDraw(True)
            
            # if action_distractor is active this frame...
            if action_distractor.status == STARTED:
                # update params
                pass
            
            # if action_distractor is stopping this frame...
            if action_distractor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > action_distractor.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    action_distractor.tStop = t  # not accounting for scr refresh
                    action_distractor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'action_distractor.stopped')
                    # update status
                    action_distractor.status = FINISHED
                    action_distractor.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in actionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "action" ---
        for thisComponent in actionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('action.stopped', globalClock.getTime())
        # Run 'End Routine' code from action_code
        for component in screen_components:
            component.setAutoDraw(False)
        
        screen_components = []
        # store data for action_loop (TrialHandler)
        action_loop.addData('action_response.x', action_response.x)
        action_loop.addData('action_response.y', action_response.y)
        action_loop.addData('action_response.leftButton', action_response.leftButton)
        action_loop.addData('action_response.midButton', action_response.midButton)
        action_loop.addData('action_response.rightButton', action_response.rightButton)
        action_loop.addData('action_response.time', action_response.time)
        action_loop.addData('action_response.clicked_name', action_response.clicked_name)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        action_loop.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            action_loop.addData('key_resp.rt', key_resp.rt)
            action_loop.addData('key_resp.duration', key_resp.duration)
        # the Routine "action" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'action_loop'
    
    
    # --- Prepare to start Routine "perception_introduction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('perception_introduction.started', globalClock.getTime())
    perception_intro_respone.keys = []
    perception_intro_respone.rt = []
    _perception_intro_respone_allKeys = []
    # keep track of which components have finished
    perception_introductionComponents = [perception_text, perception_intro_respone]
    for thisComponent in perception_introductionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "perception_introduction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *perception_text* updates
        
        # if perception_text is starting this frame...
        if perception_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            perception_text.frameNStart = frameN  # exact frame index
            perception_text.tStart = t  # local t and not account for scr refresh
            perception_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(perception_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            perception_text.status = STARTED
            perception_text.setAutoDraw(True)
        
        # if perception_text is active this frame...
        if perception_text.status == STARTED:
            # update params
            pass
        
        # *perception_intro_respone* updates
        waitOnFlip = False
        
        # if perception_intro_respone is starting this frame...
        if perception_intro_respone.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            perception_intro_respone.frameNStart = frameN  # exact frame index
            perception_intro_respone.tStart = t  # local t and not account for scr refresh
            perception_intro_respone.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(perception_intro_respone, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'perception_intro_respone.started')
            # update status
            perception_intro_respone.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(perception_intro_respone.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(perception_intro_respone.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if perception_intro_respone.status == STARTED and not waitOnFlip:
            theseKeys = perception_intro_respone.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _perception_intro_respone_allKeys.extend(theseKeys)
            if len(_perception_intro_respone_allKeys):
                perception_intro_respone.keys = _perception_intro_respone_allKeys[0].name  # just the first key pressed
                perception_intro_respone.rt = _perception_intro_respone_allKeys[0].rt
                perception_intro_respone.duration = _perception_intro_respone_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in perception_introductionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "perception_introduction" ---
    for thisComponent in perception_introductionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('perception_introduction.stopped', globalClock.getTime())
    # check responses
    if perception_intro_respone.keys in ['', [], None]:  # No response was made
        perception_intro_respone.keys = None
    thisExp.addData('perception_intro_respone.keys',perception_intro_respone.keys)
    if perception_intro_respone.keys != None:  # we had a response
        thisExp.addData('perception_intro_respone.rt', perception_intro_respone.rt)
        thisExp.addData('perception_intro_respone.duration', perception_intro_respone.duration)
    thisExp.nextEntry()
    # the Routine "perception_introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    perception_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('distractors.csv', selection=choice(268, size = 5, replace = False)),
        seed=None, name='perception_loop')
    thisExp.addLoop(perception_loop)  # add the loop to the experiment
    thisPerception_loop = perception_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPerception_loop.rgb)
    if thisPerception_loop != None:
        for paramName in thisPerception_loop:
            globals()[paramName] = thisPerception_loop[paramName]
    
    for thisPerception_loop in perception_loop:
        currentLoop = perception_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPerception_loop.rgb)
        if thisPerception_loop != None:
            for paramName in thisPerception_loop:
                globals()[paramName] = thisPerception_loop[paramName]
        
        # --- Prepare to start Routine "perception" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('perception.started', globalClock.getTime())
        # Run 'Begin Routine' code from perception_code
        random.shuffle(possible_positions)
        pos_perception_circle1, pos_perception_circle2, pos_perception_circle3, pos_perception_target = possible_positions
        target_color = choice(colors)
        distractor_color = choice(colors)
        thisExp.addData("target color", target_color)
        thisExp.addData("distractor color", distractor_color)
        
        if(target_color == "red"):
            correct_response = 'r'
        elif (target_color == "green"):
            correct_response = "g"
        elif (target_color == "blue"):
            correct_response = "b"
        else:
            print("how the fuck is target color anything other than red, green or blue")
        
        
        perception_circle1.setPos(pos_perception_circle1)
        perception_circle2.setFillColor(distractor_color)
        perception_circle2.setPos(pos_perception_circle2)
        perception_circle2.setLineColor(distractor_color)
        perception_circle3.setPos(pos_perception_circle3)
        perception_target.setFillColor(target_color)
        perception_target.setPos(pos_perception_target)
        perception_target.setLineColor(target_color)
        perception_distractor.setImage(image)
        perception_response.keys = []
        perception_response.rt = []
        _perception_response_allKeys = []
        # keep track of which components have finished
        perceptionComponents = [perception_fixation, perception_circle1, perception_circle2, perception_circle3, perception_target, perception_distractor, perception_response]
        for thisComponent in perceptionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "perception" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *perception_fixation* updates
            
            # if perception_fixation is starting this frame...
            if perception_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                perception_fixation.frameNStart = frameN  # exact frame index
                perception_fixation.tStart = t  # local t and not account for scr refresh
                perception_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                perception_fixation.status = STARTED
                perception_fixation.setAutoDraw(True)
            
            # if perception_fixation is active this frame...
            if perception_fixation.status == STARTED:
                # update params
                pass
            
            # if perception_fixation is stopping this frame...
            if perception_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_fixation.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_fixation.tStop = t  # not accounting for scr refresh
                    perception_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    perception_fixation.status = FINISHED
                    perception_fixation.setAutoDraw(False)
            
            # *perception_circle1* updates
            
            # if perception_circle1 is starting this frame...
            if perception_circle1.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_circle1.frameNStart = frameN  # exact frame index
                perception_circle1.tStart = t  # local t and not account for scr refresh
                perception_circle1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_circle1, 'tStartRefresh')  # time at next scr refresh
                # update status
                perception_circle1.status = STARTED
                perception_circle1.setAutoDraw(True)
            
            # if perception_circle1 is active this frame...
            if perception_circle1.status == STARTED:
                # update params
                pass
            
            # if perception_circle1 is stopping this frame...
            if perception_circle1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_circle1.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_circle1.tStop = t  # not accounting for scr refresh
                    perception_circle1.frameNStop = frameN  # exact frame index
                    # update status
                    perception_circle1.status = FINISHED
                    perception_circle1.setAutoDraw(False)
            
            # *perception_circle2* updates
            
            # if perception_circle2 is starting this frame...
            if perception_circle2.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_circle2.frameNStart = frameN  # exact frame index
                perception_circle2.tStart = t  # local t and not account for scr refresh
                perception_circle2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_circle2, 'tStartRefresh')  # time at next scr refresh
                # update status
                perception_circle2.status = STARTED
                perception_circle2.setAutoDraw(True)
            
            # if perception_circle2 is active this frame...
            if perception_circle2.status == STARTED:
                # update params
                pass
            
            # if perception_circle2 is stopping this frame...
            if perception_circle2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_circle2.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_circle2.tStop = t  # not accounting for scr refresh
                    perception_circle2.frameNStop = frameN  # exact frame index
                    # update status
                    perception_circle2.status = FINISHED
                    perception_circle2.setAutoDraw(False)
            
            # *perception_circle3* updates
            
            # if perception_circle3 is starting this frame...
            if perception_circle3.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_circle3.frameNStart = frameN  # exact frame index
                perception_circle3.tStart = t  # local t and not account for scr refresh
                perception_circle3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_circle3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'perception_circle3.started')
                # update status
                perception_circle3.status = STARTED
                perception_circle3.setAutoDraw(True)
            
            # if perception_circle3 is active this frame...
            if perception_circle3.status == STARTED:
                # update params
                pass
            
            # if perception_circle3 is stopping this frame...
            if perception_circle3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_circle3.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_circle3.tStop = t  # not accounting for scr refresh
                    perception_circle3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'perception_circle3.stopped')
                    # update status
                    perception_circle3.status = FINISHED
                    perception_circle3.setAutoDraw(False)
            
            # *perception_target* updates
            
            # if perception_target is starting this frame...
            if perception_target.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_target.frameNStart = frameN  # exact frame index
                perception_target.tStart = t  # local t and not account for scr refresh
                perception_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'perception_target.started')
                # update status
                perception_target.status = STARTED
                perception_target.setAutoDraw(True)
            
            # if perception_target is active this frame...
            if perception_target.status == STARTED:
                # update params
                pass
            
            # if perception_target is stopping this frame...
            if perception_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_target.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_target.tStop = t  # not accounting for scr refresh
                    perception_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'perception_target.stopped')
                    # update status
                    perception_target.status = FINISHED
                    perception_target.setAutoDraw(False)
            
            # *perception_distractor* updates
            
            # if perception_distractor is starting this frame...
            if perception_distractor.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_distractor.frameNStart = frameN  # exact frame index
                perception_distractor.tStart = t  # local t and not account for scr refresh
                perception_distractor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_distractor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'perception_distractor.started')
                # update status
                perception_distractor.status = STARTED
                perception_distractor.setAutoDraw(True)
            
            # if perception_distractor is active this frame...
            if perception_distractor.status == STARTED:
                # update params
                pass
            
            # if perception_distractor is stopping this frame...
            if perception_distractor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_distractor.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_distractor.tStop = t  # not accounting for scr refresh
                    perception_distractor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'perception_distractor.stopped')
                    # update status
                    perception_distractor.status = FINISHED
                    perception_distractor.setAutoDraw(False)
            
            # *perception_response* updates
            waitOnFlip = False
            
            # if perception_response is starting this frame...
            if perception_response.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                perception_response.frameNStart = frameN  # exact frame index
                perception_response.tStart = t  # local t and not account for scr refresh
                perception_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(perception_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'perception_response.started')
                # update status
                perception_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(perception_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(perception_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if perception_response is stopping this frame...
            if perception_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > perception_response.tStartRefresh + 8.0-frameTolerance:
                    # keep track of stop time/frame for later
                    perception_response.tStop = t  # not accounting for scr refresh
                    perception_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'perception_response.stopped')
                    # update status
                    perception_response.status = FINISHED
                    perception_response.status = FINISHED
            if perception_response.status == STARTED and not waitOnFlip:
                theseKeys = perception_response.getKeys(keyList=['r', 'g', 'b'], ignoreKeys=["escape"], waitRelease=False)
                _perception_response_allKeys.extend(theseKeys)
                if len(_perception_response_allKeys):
                    perception_response.keys = _perception_response_allKeys[-1].name  # just the last key pressed
                    perception_response.rt = _perception_response_allKeys[-1].rt
                    perception_response.duration = _perception_response_allKeys[-1].duration
                    # was this correct?
                    if (perception_response.keys == str(correct_response)) or (perception_response.keys == correct_response):
                        perception_response.corr = 1
                    else:
                        perception_response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in perceptionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "perception" ---
        for thisComponent in perceptionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('perception.stopped', globalClock.getTime())
        # check responses
        if perception_response.keys in ['', [], None]:  # No response was made
            perception_response.keys = None
            # was no response the correct answer?!
            if str(correct_response).lower() == 'none':
               perception_response.corr = 1;  # correct non-response
            else:
               perception_response.corr = 0;  # failed to respond (incorrectly)
        # store data for perception_loop (TrialHandler)
        perception_loop.addData('perception_response.keys',perception_response.keys)
        perception_loop.addData('perception_response.corr', perception_response.corr)
        if perception_response.keys != None:  # we had a response
            perception_loop.addData('perception_response.rt', perception_response.rt)
            perception_loop.addData('perception_response.duration', perception_response.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'perception_loop'
    
    
    # --- Prepare to start Routine "thank_you" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thank_you.started', globalClock.getTime())
    # keep track of which components have finished
    thank_youComponents = [thank_you_text]
    for thisComponent in thank_youComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thank_you_text* updates
        
        # if thank_you_text is starting this frame...
        if thank_you_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thank_you_text.frameNStart = frameN  # exact frame index
            thank_you_text.tStart = t  # local t and not account for scr refresh
            thank_you_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thank_you_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            thank_you_text.status = STARTED
            thank_you_text.setAutoDraw(True)
        
        # if thank_you_text is active this frame...
        if thank_you_text.status == STARTED:
            # update params
            pass
        
        # if thank_you_text is stopping this frame...
        if thank_you_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thank_you_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                thank_you_text.tStop = t  # not accounting for scr refresh
                thank_you_text.frameNStop = frameN  # exact frame index
                # update status
                thank_you_text.status = FINISHED
                thank_you_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_youComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_youComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thank_you.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)

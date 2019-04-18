#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/cred.json

# These are the pipe files we'll use
ANALYSIS_PIPE=/tmp/$1_pipe
PLAY_PIPE=/tmp/$1_play_pipe
ANNOTATION_FILE=/tmp/$1_annotations
rm -f $ANALYSIS_PIPE
rm -f $PLAY_PIPE
rm -f $ANNOTATION_FILE
echo "Creating HLS pipes"
mkfifo $ANALYSIS_PIPE
mkfifo $PLAY_PIPE
mkfifo $ANNOTATION_FILE
gst-launch-1.0 -v souphttpsrc location=$2 ! hlsdemux ! filesink location=$PLAY_PIPE &
PLAY_GST_PID=$!
gst-launch-1.0 -v souphttpsrc location=$2 ! hlsdemux ! filesink location=$ANALYSIS_PIPE &
ANALYSIS_GST_PID=$!
echo "Starting player"
echo "Start HLS Controller..."
python $(pwd)/video_player.py -v $PLAY_PIPE -l $ANNOTATION_FILE &
PLAYER_PID=$!
python $(pwd)/python/streaming_$1_tracking.py $ANALYSIS_PIPE $ANNOTATION_FILE
ANALYSIS_PID=$!
function finish {
    echo "Cleaning up!"
    # Keep these around for now for testing
    # rm -f $ANALYSIS_PIPE
    # rm -f $PLAY_PIPE
    # rm -f $ANNOTATION_FILE
    if ! kill $ANALYSIS_GST_PID > /dev/null 2>&1; then
        echo "Could not send SIGTERM to analysis GST pipe process $pid" >&2
    fi
    if ! kill $PLAY_GST_PID > /dev/null 2>&1; then
        echo "Could not send SIGTERM to play GST pipe process $pid" >&2
    fi
    if ! kill $ANALYSIS_PID > /dev/null 2>&1; then
        echo "Could not send SIGTERM to analysis python process $pid" >&2
    fi
    if ! kill $PLAYER_PID > /dev/null 2>&1; then
        echo "Could not send SIGTERM to player python process $pid" >&2
    fi
}
trap finish EXIT
sleep 1000



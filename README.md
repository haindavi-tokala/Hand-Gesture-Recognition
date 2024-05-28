Download the folder and extract it and then open it in vscode.
Initially run the collect.py which open a camera frame and capture 26 different hand gestures (which defineds the gesture for 26 letters) after pressing the letter q each time.
Then run the create.py and ignore any warnings and wait for 3 to 5 minutes till the command prompt comes back.Running this file will store all the images in serialized form in the pickle file.
After this  run the train.py which displayes the  accuracy score in the terminal.
finally run the out.py file which will lead to the camera frame along with that open a notepad next to eachother and show the hand gesture on which the model was trained and that letter should be typed in the notepad

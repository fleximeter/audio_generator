"""
File: feature_definitions.py

This module has feature maps for audio featurization.
"""

# Letter name and accidental
LETTER_NAME_ENCODING = {"C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "A": 6, "B": 7}
ACCIDENTAL_NAME_ENCODING = {"None": 0, 'double-flat': 1, 'double-sharp': 2, 'flat': 3, 'half-flat': 4, 'half-sharp': 5, 
                             'natural': 6, 'one-and-a-half-flat': 7, 'one-and-a-half-sharp': 8, 'quadruple-flat': 9, 
                             'quadruple-sharp': 10, 'sharp': 11, 'triple-flat': 12, 'triple-sharp': 13}
REVERSE_LETTER_NAME_ENCODING = {0: "None", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "A", 7: "B"}
REVERSE_ACCIDENTAL_NAME_ENCODING = {0: "None", 1: 'double-flat', 2: 'double-sharp', 3: 'flat', 4: 'half-flat', 
                                     5: 'half-sharp', 6: 'natural', 7: 'one-and-a-half-flat', 8: 'one-and-a-half-sharp', 
                                     9: 'quadruple-flat', 10: 'quadruple-sharp', 11: 'sharp', 12: 'triple-flat', 13: 'triple-sharp'}

###################################################################################################################
# The total number of features and outputs for the model. This can change from time to time, and must be updated!
###################################################################################################################
# NUM_FEATURES = len(LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(QUARTER_LENGTH_ENCODING) + len(BEAT_ENCODING) + \
#                len(PITCH_CLASS_ENCODING) + len(MELODIC_INTERVAL_ENCODING) + len(KEY_SIGNATURE_ENCODING)  + \
#                len(MODE_ENCODING) + len(TIME_SIGNATURE_ENCODING)
# NUM_OUTPUTS = len(LETTER_ACCIDENTAL_OCTAVE_ENCODING) + len(QUARTER_LENGTH_ENCODING)

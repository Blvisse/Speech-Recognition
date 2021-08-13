import unittest
import pandas as pd
import sys,os
import librosa 

sys.path.append(os.path.abspath(os.path.join('../..')))


# import dvc.api


class TestsAB(unittest.TestCase):


    
    
    
    def test_sample(self):


        samples,sample_rate=librosa.load('../scripts/stereo.wav')
        print(sample_rate)

        

        self.assertEqual(sample_rate,22050)

    
    def test_sample_funct(self):

        samples,sample_rate=librosa.load('../scripts/newsound.wav')
        print(sample_rate)
        
        samples,sample_rate=librosa.load('../scripts/newsound.wav',sr=8000)
        print(sample_rate)


        self.assertEqual(sample_rate,8000)


   

if __name__ == '__main__':
    unittest.main()


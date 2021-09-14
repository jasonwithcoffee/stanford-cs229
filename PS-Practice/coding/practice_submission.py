import matplotlib
import numpy
import scipy
import pandas as pd

class PracticeSubmission(object): 
    def __init__(self, response=None):
        """
        Args:
            response: A string
        """
        self.response = response
    
    def returnResponse(self):
        self.response = "I successfully completed the conda tutorial!"
        return self.response

def runPracticeSubmission(response):
    practice = PracticeSubmission()
    response = practice.returnResponse()
    return response

def main(response):
    response = runPracticeSubmission(response)

if __name__ == '__main__':
    main("Try this out!")

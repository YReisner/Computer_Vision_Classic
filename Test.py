"""
This is a test
This also
"""

Params= GetDefaultParameters()

DandL= GetData(Params[‘Data’])

SplitData= TrainTestSplit( DandL[‘Data’], DandL[‘Labels’], Params[‘Split’])
				 # returns train data, test data, train labels and test labels

TrainDataRep = prepare( SplitData[‘Train’][‘Data’], Params[‘Prepare’])

Model =  Train(TrainDataRep, SplitData[‘Train’][‘Labels’] , Params[‘Train’])
TestDataRep = Prepare(SplitData[‘Test’][‘Data’], Params[‘Preapare’])

Results = Test(Model, TestDataRep)

Summary = Evaluate(Results, SplitData[‘Test’][‘Labels’], Params[‘Summary’])

ReportResults(Summary, Params[‘Report’])

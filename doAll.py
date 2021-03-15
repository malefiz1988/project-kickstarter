import dataCleaning
#import featureEngineering
#import logRegModelCreate
#import knnModelCreate
#import naiveBayesModelCreate
#import decisionTreeModelCreate
#import svcModelCreate
#import randomForestModelCreate
#import adaBoostModelCreate
import modelNoStaffNoBackersAB
import modelStaffRF
import modelBackerStaffRF
import modelBackerRF


dataCleaning.main()


#logRegModelCreate.main()
#knnModelCreate.main()
#naiveBayesModelCreate.main()
#decisionTreeModelCreate.main()
#svcModelCreate.main()
#randomForestModelCreate.main()
#adaBoostModelCreate.main()

modelNoStaffNoBackersAB.main()
modelStaffRF.main()
modelBackerStaffRF.main()
modelBackerRF.main()
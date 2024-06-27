import utilsCamLogs as cutils
from importlib import reload
reload(cutils)
from datetime import datetime, timedelta
import json
import numpy as np
import matplotlib.pyplot as plt

def makeTimePlots(gres, mapCamVidPlateAc):
    tags = {}
    for n in range(len(gres)):
        cname = gres[n]['cam']
        ent = mapCamVidPlateAc[cname]
        tags[cname] = str(ent['plate'])+','+ str(ent['vid'])
    cutils.plotCamTrips(gres, 'latestPlots', cutils.baseDir+'report/', tags=tags)

def sanitizeLabel(label):
    label = label[0].upper()+label[1:]
    label = label.replace('Bad','Poor')
    if label.count('.'):
        ls = label.split('.')
        for n in range(len(ls)):
            ls[n] = ls[n][0].upper() + ls[n][1:]
        label = '.'.join(ls)
    return label

def sanitizeLabels(labels):
    for n in range(len(labels)):
        labels[n] = sanitizeLabel(labels[n])
    return labels

def makePiPlot(camSumm, logName):
    print('make pi plot...')
    labels = list(camSumm.columns)
    y = []
    for lab in labels:
        y.append(camSumm.iloc[-1][lab])

    if 'ok' in labels: #merge ok with good
        n = labels.index('ok')
        m = labels.index('good')
        y[m] += y[n]
        y.pop(n)
        labels.pop(n)
        
    labels = sanitizeLabels(labels)
    cols = []
    colLeg = {'Good':'green', 'No trips':'orange', 'SD issue':'pink', 'Poor':'red', 'OFF':'lightblue'}
    for n in range(len(labels)):
        cols.append(colLeg[labels[n]])
        labels[n] += ' ('+str(y[n])+')'
        
    plt.pie(y, labels = labels, colors=cols)
    plt.title(logName)
    plt.tight_layout()
    plt.savefig(cutils.baseDir+ 'report/latestStatus.png')
    plt.savefig(cutils.baseDir+ 'logs/latestStatus_%s.png'%logName)

    
def getDFByState(camSumm):
    import pandas as pd
    col = 'state'
    df = camSumm[['name',col]]
    grps = {}
    for n in range(len(df)):

        k = df.iloc[n][col]
        k = k.replace('no trips','no.trips').replace('ok','good').replace('SD issue','SD.issue').split(' ')[0]

        if k not in grps:
            grps[k] = []
        grps[k].append(df.iloc[n]['name'])
    grpDF = pd.DataFrame()
    for grp in grps:
        ldf = pd.DataFrame(grps[grp])
        ldf.columns = [grp]
        grpDF = pd.merge(grpDF, ldf, left_index=True, right_index=True, how='outer')
    grpDF.replace(np.nan,'')
    cols = list(grpDF.columns)
    cols = sanitizeLabels(cols)
    grpDF.columns = cols
    return grpDF

def makeHtmlPage(camMetrics):
    print('make HTML page ...')
    grpDF = getDFByState(camMetrics)
    
    htxt = '''
    <html><head>
    <style>
    * {
      font-family: arial;
    }
    td,tr,table{
      border-collapse: collapse;
      padding: 5px;
      padding-left: 10px;
      padding-right: 10px;
      text-align: center;
    }
    </style>
    </head><body>
    '''

    htxt += '<h1>Intangles Camera Status:  '+logName+'</h1>'
    htxt += '<img src="latestStatus.png">'
    htxt += grpDF.to_html(na_rep="", justify='center')

    htxt+= '</body></html>'

    with open(cutils.baseDir+'report/latest.html','w') as f:
        f.write(htxt)
        
mapCamVidPlateAc = json.load(open(cutils.baseDir+'data/cam2VidPlateAc.json','r'))
selCNames = list(mapCamVidPlateAc.keys())

import sys
if __name__ == '__main__':
    cutils.log('started run ', 'w')

    if len(sys.argv)>1:
        n = int(sys.argv[1])
        if n>0:
            selCNames = selCNames[:n]
            print(selCNames)
    
    gmtDelta = cutils.timedelta(seconds=5*3600+30*60)
    tim = cutils.datetime.now()
    en = tim.replace(hour=15, minute=0, second=0, microsecond=0)
    en = en - gmtDelta
    logName = en.strftime('%d%b')
    cutils.log(logName+' no of cams=%d'%len(selCNames))
    print(logName, en)
    offstHrs = 5

    if True:
        gres = []
        gres = cutils.getCamLogsByTrip(selCNames, en=en, offstHrs=offstHrs, gres=gres)

        cutils.log('getCamMetrics ...')
        camMetrics, camSumm = cutils.getCamMetrics(gres, logName, cutils.baseDir+'logs/', mapCamVidPlateAc)

        cutils.log('makeTimePlots ...')
        makeTimePlots(gres, mapCamVidPlateAc)
    
    else:
        import pandas as pd
        camMetrics = pd.read_csv(cutils.baseDir+'logs/df_%s.csv'%logName, index_col=0)
        camSumm = pd.read_csv(cutils.baseDir+'logs/summ_%s.csv'%logName, index_col=0)
    
    cutils.log('makePiPlot ...')
    makePiPlot(camSumm, logName)

    cutils.log('makeHtmlPage ...')
    makeHtmlPage(camMetrics)

    cutils.log('ALL DONE')

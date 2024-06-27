import requests 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pandas as pd
import os, sys, glob, json
from datetime import datetime
from datetime import timedelta

baseDir = '/home/ubuntu/vikram/CameraStatus/'
logFile = baseDir+'log.txt'

brahmaURL = 'https://apis.intangles.com/idevice/logsV2/{cname}?psize=10000&token=JBRHjp1tPFdyZwvRGblGwi-hIv5OmYu-cr--qzRE9rSCY6F1M5vxQt5Y7Wn9g7ur&'
mapCamVidPlateAc = json.load(open(baseDir+'data/cam2VidPlateAc.json','r'))

def log(msg, mode='a'):
    with open(logFile, mode) as f:
        f.write(str(datetime.now())+':'+msg+'\n')

gURL = None
gResp = None
def getCamBrahmaLogs(cname, en=None, offstHrs=2, st=None):
    # downloads logs from Brahma and returns logs and response
    
    global gURL, gResp
    # offstHrs = None will return the last logs
    # en = None will set end time to now()
    
    url = brahmaURL.format(cname=cname)
    if offstHrs is not None:
        if en is None:
            en = datetime.now()
        if type(en) is datetime:
            en = int(en.timestamp()*1000)    
        st = int(en - offstHrs*60*60*1000)
        st = str(st)    
        en = str(en)
        url = url + '&from='+st+'&until='+en #+'&last_t='+str(int(st)+1000)
        
    gURL = url
    gResp = requests.get(url)
    if gResp.status_code==200:
        logs = gResp.json()['logs']
    else:
        print('\t',cname,': brahma returned unusual status code',gResp.status_code)
        logs = []
    return logs, gResp

def getCamBrahmaLogsBySlot(cname, en, st):
    # downloads logs from Brahma and returns logs and response
    
    global gURL, gResp
    # offstHrs = None will return the last logs
    # en = None will set end time to now()
    
    url = brahmaURL.format(cname=cname)
    st = str(int(st.timestamp()*1000))    
    en = str(int(en.timestamp()*1000))    
    url = url + '&from='+st+'&until='+en #+'&last_t='+str(int(st)+1000)

    gURL = url
    gResp = requests.get(url)
    if gResp.status_code==200:
        logs = gResp.json()['logs']
    else:
        print('\t',cname,': brahma returned unusual status code',gResp.status_code)
        logs = []
    return logs, gResp

def getCamBrahmaLogsLong(cname, en, offstHrs, getLastLog=True, dbg=False):
    # download logs for a long time (by repeated downloads), returns logs
    st = en - timedelta(seconds=offstHrs*60*60)
    cen = en
    ncnt = 0
    flogs = []
    while(cen>st):
        ncnt += 1
        if dbg:
            print(ncnt,'getCamBrahmaLogsBySlot',cname,cen,st)
        logs, resp = getCamBrahmaLogsBySlot(cname, en=cen, st=st)
        if len(logs)==0:
            if len(flogs)==0 and getLastLog:
                flogs, resp = getCamBrahmaLogs(cname, offstHrs=None)
            break
            
        flogs += logs
        cen = datetime.fromtimestamp(logs[-1]['t']/1000)
    return flogs

camStatusDesc = {'0': 'keep alive', '1': 'connected', '2': 'disconnected', '3':'end session',
                 '4':'heartbeat', '5':'igntn on','4.5':'alerts', '4.25':'hbt parse error', '1.5':'server,connecting'}

def getCamBrahmaEntries(logs):
    # parse log and returns entries
    
    ents = []
    for nlog, log in enumerate(logs):
        t = datetime.fromtimestamp(log['t']/1000)
        stat = -1
        if log['a'].find('keep_alive')>=0:
            stat = 0
        elif log['a'].find('published')>=0:
            if log['topic'].find('heartbeat')>=0:
                hbt = log['m']
                stat = 4
                try:
                    pk = json.loads(hbt)
                    if 'ignition' in pk:
                        if pk['ignition']:
                            stat = 5                
                except:
                    stat = 4.25
                    pass
            elif log['topic'].find('end_session')>=0:
                stat = 3
            else:
                stat = 4.5
        elif log['a'].find('connected')>=0:
            stat = 1
        elif log['a'].find('disconnected')>=0:
            stat = 2
        elif log['a'].find('server')>=0 or log['a'].find('connecting')>=0:
            stat = 1.5        
#         else:
#             print(log)
        ents.append([t,stat])
    ents = np.array(ents)
    return ents

def getCamSessions(brahmaEnts, dbg=False):
    # parse entries and returns sessions [not update, dont use]
    
    sessions = []
    sessOn = 0
    tSess = []
    brkCnt = 0
    brkThresh = 5
    pt = brahmaEnts[0,0]
    for nent, ent in enumerate(brahmaEnts):
        ct = ent[0]
        timeBrk = False
        if (pt-ct).total_seconds()>60:
            timeBrk = True
        pt = ct
        
        if ent[1]>=4 and (not timeBrk):
            sessOn = 1
        else:
            brkCnt += 1 # breakcount to avoid momentary dips
            if brkCnt>brkThresh or timeBrk:
                sessOn = 0
                brkCnt = 0
                if len(tSess)>10:
                    sessions.append((tSess[0][1][0],tSess[-1][1][0],len(tSess),tSess[0][0],tSess[-1][0]))
                tSess = []

        if sessOn:
            tSess.append((nent,ent))
        
    if sessOn:
        if len(tSess)>10:
            sessions.append((tSess[0][1][0],tSess[-1][1][0],len(tSess),tSess[0][0],tSess[-1][0]))
    return sessions

def getBrahmaData(cname, en=None, offstHrs=2, getLastLog=True):
    # download, process cam logs from Brahma to return entries, sessions, logs
    
    # logs, resp = getCamBrahmaLogs(cname, en, offstHrs)
    logs = getCamBrahmaLogsLong(cname, en, offstHrs, getLastLog)
    ents = getCamBrahmaEntries(logs)
    # sessions = getCamSessions(ents)
    sessions = []
    return ents, sessions, logs

def combineSessions(sessions, threshSec=60):
    # combines sessions [not updated, dont use]
    if len(sessions)==0:
        return []
    msess = [list(sessions[0])]
    for n in range(1,len(sessions)):
        dt = msess[-1][1] - sessions[n][0]
        if dt.total_seconds()< threshSec:
            msess[-1][1] = sessions[n][1]
        else:
            msess.append(list(sessions[n]))
    return msess

def plotCamTripsPlt(cname, ents, sessions, ptrips, tag=""):
    n1 = 0
    n2 = len(ents)
    plt.figure(figsize=(10,3))
    plt.plot(ents[n1:n2,0], ents[n1:n2,1], '-x')
    if len(ptrips)>0 and len(sessions)>0:
        for trip in ptrips:
            plt.vlines(trip[0],0,3,'b')
            plt.vlines(trip[1],0,3,'r')
            plt.hlines(3,trip[0],trip[1],'k')
    plt.grid('on')
    plt.title(cname+',t:%d'%len(ptrips)+','+tag)


def plotlyForceShow():
    pyo.init_notebook_mode(connected=True)

def plotlyPlot(x, primary, secondary=None, segLists=None, title='', show=False, mode='lines'):
    # x axis values
    # primary as a dict {'name: array}
    # secondary as a dict {'name: array}
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    for nam in primary:
        val = np.array(primary[nam])
        fig.add_trace(
            go.Scatter(x=x, y=val, name=nam, mode=mode, marker_size=4),
            secondary_y=False,
        )
    if secondary is not None:
        for nam in secondary:            
            val = np.array(secondary[nam])
            fig.add_trace(
                go.Scatter(x=x, y=val, name=nam, mode=mode, marker_size=4),
                secondary_y=True,
            )
    if segLists is not None:
        cols = ['#ff0000','#000000','#0000ff']
        for nseg, segs in enumerate(segLists):
            for seg in segs[2]:
                xs = [seg[0], seg[1]]
                ys = [segs[1], segs[1]]
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, name=segs[0], mode='lines+markers', marker_size=4, line=dict(color=cols[nseg])),
                    secondary_y=False,
                )
            
    config = {'responsive': False}
    fig.update_layout(title=dict(text=title))
    fig.update_layout(showlegend=False)
    if show:
        fig.show(config=config)
    return fig

def plotCamTrips(gres, logName, outDir='./', tags={}):
    with open(os.path.join(outDir,'%s.html'%logName), 'w') as f:
        for res in gres:
            ents = res['ents']
            ptrips = res['trips']
            nam = res['cam']

            if len(ents)==0:
                print(nam,'no Brahma data or trips',len(ents),len(ptrips))
                continue

            segLists = []
            segLists.append(('trip',2.75,res['trips']))
            
            title = res['cam']
            if res['cam'] in tags:
                title += ','+tags[res['cam']]
            fig = plotlyPlot(x=ents[:,0], primary={'cam':ents[:,1]}, segLists=segLists, title=title)
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

def trip_data_download(vehicle_id, start_ts, end_ts):
    trip_url = 'http://internal-apis.intangles.com/trips/'+ str(vehicle_id) + '/gettripsintime/' + str(start_ts) + '/' + str(end_ts) + '?proj=_id,start_time,end_time'
    # getting the trip-data
    trip_response = requests.get(trip_url, json=trip_url)
    if trip_response.status_code == 200:
        # Parse the JSON response
        trip_json_data = trip_response.json()
        return trip_json_data['result']
    return []

def getCamTrips(cname, st=None, en=None):
    en = int(en.timestamp()*1000)
    st = int(st.timestamp()*1000)
    vid = mapCam2Vid[cname]
    trips = trip_data_download(vid, st, en)
    return trips

def getCamLogsByTrip(selCNames, en, offstHrs=3, byTrip=True, gres=[]):
    st  = en - timedelta(seconds=offstHrs*3600)
    step = int(st.timestamp()*1000)
    enep = int(en.timestamp()*1000)

    print('st:',st,' -- en:',en)
    log('getCamLogsByTrip: st='+str(st)+' -- en='+str(en))
    for ncam, cname in enumerate(selCNames):
        status = ''
        # vid = mapCam2Vid[cname]
        vid = mapCamVidPlateAc[cname]['vid']
        print('%5d/%-5d'%(ncam+1, len(selCNames)), cname, vid)
        log('%5d/%-5d cname=%s vid=%s'%(ncam+1, len(selCNames), cname, vid))

        if byTrip:
            # 1. get trips
            for n in range(3):
                trips = trip_data_download(vid, step, enep)
                if len(trips)>0:
                    break

            ptrips = []
            for trip in trips:
                if 'end_time' not in trip:
                    continue
                ptrips.append( (datetime.fromtimestamp(trip['start_time']/1000), datetime.fromtimestamp(trip['end_time']/1000)) )

            if len(ptrips)==0:
                status = 'no trips'
                logs, resp = getCamBrahmaLogs(cname, en=en, offstHrs=None)
                ents = getCamBrahmaEntries(logs)
                print('\t',cname, status)
                gres.append({'cam':cname, 'vid':vid, 'ents':ents,'logs':logs, 'trips':ptrips,'st':st,'en':en})
                continue
            print('\t','no. of trips:',len(ptrips))

            # 2. get cam logs
            # 2a. add 20 min buffer
            cen = ptrips[0][1] + timedelta(seconds=10*60)
            cst = ptrips[-1][0] - timedelta(seconds=10*60)
            coffst = np.ceil((cen - cst).total_seconds()/3600)
            cst = cen - timedelta(seconds=coffst*3600)
        else:
            cen = en
            coffst = offstHrs
            ptrips = []

        # 2b. get Brahma logs
        print('\tget brahma logs:',cst,' -- ',cen)
        ents, sessions, logs = getBrahmaData(cname, en=cen, offstHrs=coffst, getLastLog=False)
        if len(ents)==0:
            status = 'no Brahma data'
            print('\t',cname, status)    
        gres.append({'cam':cname, 'vid':vid, 'ents':ents,'logs':logs, 'trips':ptrips,'st':st,'en':en,'cen':cen,'coffst':coffst})
    log('done with getCamLogsByTrip')
    return gres

def getCamMetrics(gres, logName, outDir='', metaInfo={}):
    
    camMetrics = []
    for res in gres:
        ents = res['ents']
        trips = res['trips']
        
        if len(ents)==0:
            camMetrics.append((res['cam'], -1, -1, -1, len(trips), -1, -1, ''))
            continue
        
        def sdCardChk():
            txt = json.dumps(res['logs'])
            isd = txt.find('sdCardInfo')
            if isd<0:
                return 0
            sdn = int(txt[isd:isd+100].split('status\\":')[1].split('}')[0])
            if sdn==2:
                return 1
            else:
                return 0
            
        if len(trips)==0:
            if len(ents)==0:
                camMetrics.append((res['cam'],-1,-1,-1,len(trips),-1,-1,''))
            else:
                camMetrics.append((res['cam'],-1,-1,-1,len(trips),-1,sdCardChk(),'%d'%len(ents)))
            continue
    
        segLists = []
        evOk = np.ones(len(ents), dtype=int)
        evOk[ents[:,1]<4] = -1 # to detect how many times camera was running outside of trips
        perOks = []
        for ntrip, trip in enumerate(trips):
            st = trip[0]
            en = trip[1]
            missTrip = 0
            relEntsIndx = np.where((ents[:,0]>=st)*1 * (ents[:,0]<=en)*1 * (ents[:,1]>=4)*1)[0]        
            if len(relEntsIndx)<1:
                perOk = 0
            else:
                pev = ents[relEntsIndx[0],0]
                camDur = 0
                lev = pev
                for nk, k in enumerate(relEntsIndx[1:]):
                    cev = ents[k,0]
                    gap = (pev-cev).total_seconds()
                    if gap>300:
                        #sess.append((pev, lev))
                        elap = (lev-pev).total_seconds()
                        camDur += elap
                        lev = cev
                    pev = cev
                # sess.append((lev,pev))
                camDur += (lev-pev).total_seconds()
                # print(relEntsIndx, trip)
                evOk[relEntsIndx] = False
                tripDur = (trip[1]-trip[0]).total_seconds()
                perOk = camDur*100.0/tripDur
            perOks.append(perOk)
    
        nCamOn = len(np.where(evOk>=0)[0])
        nXtraVid = len(np.where(evOk>0)[0]) 
        camOverPer = nXtraVid*100.0 / nCamOn if nCamOn>0 else -1 # % instance camera ran outside of trips
        goodTrips = np.sum(np.array(perOks)>80) # count of trips with cam overlap > 80%
        goodTripsPer = goodTrips*100.0/len(perOks) # % trips with cam overlap >80%
        metrics = (res['cam'], np.round(np.mean(perOks),0), np.round(goodTripsPer,0), goodTrips, len(trips), 
                   np.round(camOverPer,0), sdCardChk(), '%d|%d|%d'%(len(evOk),nCamOn,nXtraVid))
        camMetrics.append(metrics)
        
    df = pd.DataFrame(camMetrics)
    statCols = ['avg.perOk','goodTrips(%)','goodTrips','#trips','camOveruse(%)','SD']
    df.columns = ['name'] + statCols + ['cnts']

    states = []
    for n in range(len(df)):
        ent = df.iloc[n]
        if ent['SD']>0:
            state = 'SD issue'
        elif ent['#trips']==0:
            state = 'no trips'
        elif ent['cnts'] is None:
            state = 'no logs'
        elif len(ent['cnts'])==0:
            state = 'no logs'
        elif ent['#trips']==0:
            state = 'no trips'
        else:
            if ent['avg.perOk']>80:
                if ent['camOveruse(%)']>30:
                    state = 'ok'
                else:
                    state = 'good'
            elif ent['avg.perOk']<20:
                state = 'poor'
            else:
                state = 'ok'
        if state.find('no logs')>=0 and ent['#trips']>0:
            state = 'OFF'
        states.append(state)
    df['state'] = states
    
    vids = []
    plates = []
    for n in range(len(df)):
        cname = df.iloc[n]['name']
        if cname in metaInfo:
            vids.append(metaInfo[cname]['vid'])
            plates.append(metaInfo[cname]['plate'])
        else:
            vids.append('NA')
            plates.append('NA')
    df['vid'] = vids
    df['plate'] = plates
    
    df.to_csv(os.path.join(outDir,'df_%s.csv'%logName))
    camMetricsDF = df.sort_values('state')
    # display(camMetricsDF)
    
    summDF = pd.DataFrame()
    for state in df.state.unique():
        ldf = df[df.state==state]
        lsumm  = ldf[statCols].mean(axis=0)
        lsumm['count'] = len(ldf)
        lsumm = lsumm.astype(int)
        lsumm.name = state
        summDF = pd.concat([summDF, lsumm],axis=1)
    # display(summDF.iloc[-1:])
    summDF.to_csv(os.path.join(outDir,'summ_%s.csv'%logName))

    return camMetricsDF, summDF

def compareMetrics(outDir):
    
    gdf = None
    fils = glob.glob( os.path.join(outDir,'df_*.csv') )
    fils = list(zip(fils,[x.split('_')[1].split('.')[0] for x in fils]))
    fils.sort(key=lambda x: datetime.strptime('2024 '+x[1], '%Y %d%b'))
    for fil,nam in fils:
        ldf = pd.read_csv(fil)
        if 'state' not in ldf.columns:
            continue
        sagg = []
        for n in range(len(ldf)):
            x = ldf.iloc[n]
            sagg += [x['state']+' (%2d,%2d)'%(x['#trips'],x['SD'])]
        ldf = ldf[['name','state']]
        ldf['state'] = sagg
        ldf.columns = ['name', nam]
        if gdf is None:
            gdf = ldf
        else:
            gdf = pd.merge(gdf, ldf, left_on='name', right_on='name')
    # display(gdf)
    return gdf

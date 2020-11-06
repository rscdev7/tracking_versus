/*
@author :   rscalia
@date   :   Wed 29/07/2020

Questo componente serve per elaborare un filmato con l'algoritmo TLD.

*/

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "./lib/libDataManager.hpp"
#include "imgStream.h"
#include <TLD.h>


using namespace std;
using namespace cv;
using namespace DTM;


void configureTracker(tld::TLD *tracker,Mat grey);
void selectTarget (DataManager pDm);


//CONFIG
cv::Mat imgDisplay;
cv::Rect initRect;
string DATA_PATH                = "./data/gt.txt";
string WRITE_PATH               = "./data/detections.txt";
float  THRESHOLD                = 1.0;
string SYSTEM                   = "WH";
int    LIMIT                    = DTM::EMPTY_INTEGER;
string INPUT_VIDEO              = "./data/vid.mp4";
string OUTPUT_VIDEO             = "./data/vid_out.mp4";
int    FIRST_FRAME_TO_TRACK     = -1;
int    IDX_TO_TRACK             = -1;


int main() {

    //Variabili di Init
    int seed        = 0;
    srand(seed);
    int method      = IMACQ_VID;


    //Inizializzo imgStream e DataManager
    DataManager c   = DataManager(DATA_PATH,THRESHOLD, LIMIT, SYSTEM, WRITE_PATH);
    c.readData();

    imgStream       *test;
    test = new imgStream(method , INPUT_VIDEO);

    tld::TLD *trackerTLD;

    
    //Scelta Target
    selectTarget(c);
    
    //Costruzione oggetto per la generazione del video di output
    Size S      = Size((int) test->capture->get(CV_CAP_PROP_FRAME_WIDTH),    
                  (int) test->capture->get(CV_CAP_PROP_FRAME_HEIGHT));
    int fps     = test->capture->get(CV_CAP_PROP_FPS);

    VideoWriter video(OUTPUT_VIDEO,CV_FOURCC('F','M','P','4'), fps, S);
    
    
    cv::Mat imageCurr;
    int i = 0;

    while (test->getCurrImage() == 1) {

        i++;

        cout<<"Immagine Elaborata N "<<i<<" \n"<<endl;

        
        test->currImage.copyTo(imageCurr);
        imageCurr.copyTo(imgDisplay);

        if (i == FIRST_FRAME_TO_TRACK) {

            Mat grey(imgDisplay.rows,imgDisplay.cols, CV_8UC1);
            cvtColor(imgDisplay,grey,CV_BGR2GRAY);

            trackerTLD = new tld::TLD();
            configureTracker(trackerTLD,grey);
            trackerTLD->selectObject(grey,imgDisplay,&initRect);

            //LOGGING
            vector<float> arr (4);
            arr[0] = (float)trackerTLD->currBB->x;
            arr[1] = (float)trackerTLD->currBB->y;
            arr[2] = (float)trackerTLD->currBB->width;
            arr[3] = (float)trackerTLD->currBB->height;

            c.writeLine(i, IDX_TO_TRACK, arr, SYSTEM);

        }else if (i > FIRST_FRAME_TO_TRACK) {

            //Processing Frame
            trackerTLD->processImage(imageCurr);

    
            if (trackerTLD->currBB != NULL) {

                //Stampa BBOX
                cv::rectangle(imgDisplay,Rect(trackerTLD->currBB->x,trackerTLD->currBB->y,trackerTLD->currBB->width,trackerTLD->currBB->height),Scalar(0,255,0,255),4);

                //LOGGING
                vector<float> arr (4);
                arr[0] = (float)trackerTLD->currBB->x;
                arr[1] = (float)trackerTLD->currBB->y;
                arr[2] = (float)trackerTLD->currBB->width;
                arr[3] = (float)trackerTLD->currBB->height;

                c.writeLine(i, IDX_TO_TRACK, arr, SYSTEM);
            }
        }
        

        //Write del video di output
        video.write(imgDisplay);
    }


    //Libero le risorse allocate
    delete trackerTLD;
    delete test;
    return 0;
}


void configureTracker(tld::TLD *tracker , Mat grey) {
    tracker->alternating = false;
    tracker->trackerEnabled = true;
    tracker->learningEnabled = true;
    tracker->detectorCascade->varianceFilter->enabled = true;
    tracker->detectorCascade->ensembleClassifier->enabled = true;
    tracker->detectorCascade->nnClassifier->enabled = true;

    // classifier
    tracker->detectorCascade->useShift = true;
    tracker->detectorCascade->shift = 0.1;
    tracker->detectorCascade->minScale = -10;
    tracker->detectorCascade->maxScale = 10;
    tracker->detectorCascade->minSize = 25;
    tracker->detectorCascade->numTrees = 10;
    tracker->detectorCascade->numFeatures = 13;
    tracker->detectorCascade->nnClassifier->thetaTP = 0.65;
    tracker->detectorCascade->nnClassifier->thetaFP = 0.5;

    tracker->detectorCascade->imgWidth = grey.cols;
    tracker->detectorCascade->imgHeight = grey.rows;
    tracker->detectorCascade->imgWidthStep = grey.step;
}


void selectTarget (DataManager pDm) {

    cout<<"\n****************DATA INFO********************"<<endl;
    cout<<"-> Data Path: "<<pDm._dataPath<<endl;
    cout<<"-> Limit: "<<pDm._limit<<endl;
    cout<<"-> System: "<<pDm._system<<endl;
    cout<<"-> Write Path:"<<pDm._writePath<<endl;
    cout<<"********************************************\n"<<endl;


    //Prelievo Targets
    pDm.getAllTargets();
    int target_sel = -1;
    map<int, int>::iterator ls = pDm._avaibleTargets.begin();


    //Visualizzazione Target disponibili
    cout<<"\n[!] Target Disponibili: \n"<<endl;
    cout<<"[ ";
    while (ls != pDm._avaibleTargets.end()) {
        int  key   = ls->first;
        
        ls++;


        if (ls == pDm._avaibleTargets.end()) {
            cout<<key;
        }else{
            cout<<key<<" , ";
        }
    }
    cout<<" ]\n"<<endl;


    //Selezione Target
    cout<<"[!] Seleziona un target \n"<<endl;
    cin>>target_sel;


    //Notifica Avvenuta selezione
    map<int, int>::iterator it ;
    it = pDm._avaibleTargets.find(target_sel);

    if(it == pDm._avaibleTargets.end()) {
        cout<<"\n[#] Errore, inserito un target inesistente \n"<<endl;
        return;
    }else{
        cout<<"\n-> È stato selezionato il target con ID = "<<target_sel<<" - Primo Frame Traiettoria "<<it->second<<" \n"<<endl;
        IDX_TO_TRACK  = target_sel;
    }


    //Memorizzazione Selezione
    struct MapId key;
    key.frameNumber                 = it->second;
    FIRST_FRAME_TO_TRACK            = key.frameNumber;
    key.idfFrame                    = it->first;
    map<MapId, Record>::iterator w  = pDm._data->find(key);

    if(w == pDm._data->end()) {
        cout<<"\n[#] Errore nel prelievo del target dalla struttura dati \n"<<endl;
        return;
    }else{
        initRect = Rect(w->second.x_1 , w->second.y_1 ,w->second.x_2, w->second.y_2);

        cout<<"[!] BBox Selected: [ "<<initRect.x<<" , "<<initRect.y<<" , "<<initRect.width<<" , "<<initRect.height<<" ]\n"<<endl;
    }

    

}
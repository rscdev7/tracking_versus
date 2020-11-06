/*
@author :   rscalia
@date   :   Wed 22/07/2020

Questo componente serve per la lettura della Bounding-Box di Ground-Truth dal file di testo.

Inoltre, il componente permette la scrittura su file delle ipotesi generate dal tracker.

*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <map>
#include <fstream>
#include <set> 
#include <iterator>

using namespace std;

namespace DTM {
    const int EMPTY_INTEGER = -1;
    const int EMPTY_DOUBLE  = -1.0;

    struct MapId {
        int frameNumber;
        int idfFrame;

        bool operator==(const MapId &o) const {
            return frameNumber == o.frameNumber && idfFrame == o.idfFrame;
        }

        bool operator<(const MapId &o)  const {
            return frameNumber < o.frameNumber || (frameNumber == o.frameNumber && idfFrame < o.idfFrame);
        }
    };

    struct Record {
        int x_1;
        int y_1;
        int x_2;
        int y_2;
    };

    class DataManager{
       
        public:
            string                _dataPath;
            map<MapId,Record>*    _data;
            float                 _threshold;
            int                   _limit;
            string                _system;
            string                _writePath;
            map<int,int>          _avaibleTargets;

        DataManager(string pDataPath, float pThreshold, int pLimit, string pSystem, string pWritePath);

        void printValues(); 
        void readData();
        void writeLine (int pFrameNumber, int pIdx, vector<float> pBbox, string pEmplSystem);
        void getAllTargets ();

    };
    
}   




/*
@author :   rscalia
@date   :   Wed 22/07/2020

Questo componente serve per testare la classe DataManager.

*/

#include <iostream>
#include "../lib/libDataManager.hpp"

using namespace std;
using namespace DTM;

//Config
string DATA_PATH   = "../data/gt.txt";
string WRITE_PATH  = "../data/detections.txt";
float  THRESHOLD   = DTM::EMPTY_DOUBLE;
string SYSTEM      = "Inc";
int    LIMIT       = DTM::EMPTY_INTEGER;


int main() {
    DTM::DataManager c = DTM::DataManager(DATA_PATH,THRESHOLD, LIMIT, SYSTEM, WRITE_PATH);

    //Test Funzione Lettura Dati
    c.readData();

    
    //Test Funzione Scrittura su File
    vector<float> arr (4);
    arr[0] = 50.0;
    arr[1] = 200.0;
    arr[2] = 50.0;
    arr[3] = 70.0;

    c.writeLine(50,20,arr,"Inc");


    //Test Funzione GetAllTargets
    c.getAllTargets();


    //Test Funzione Stampa Dati
    c.printValues();
}
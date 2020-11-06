/*
@author :   rscalia
@date   :   Wed 22/07/2020

Questo componente serve per la lettura della Bounding-Box di Ground-Truth dal file di testo.

Inoltre, il componente permette la scrittura su file delle ipotesi generate dal tracker.

*/

#include "libDataManager.hpp"
using namespace DTM;

      
DTM::DataManager::DataManager(string pDataPath, float pThreshold, int pLimit, string pSystem, string pWritePath){
    this->_dataPath         = pDataPath;
    this->_threshold        = pThreshold;
    this->_limit            = pLimit;
    this->_system           = pSystem;
    this->_writePath        = pWritePath;
    this->_data             = new map<MapId,Record>();
}

void DTM::DataManager::printValues() {
    cout<<"********************************************"<<endl;
    cout<<"-> Data Path: "<<this->_dataPath<<endl;
    cout<<"-> Limit: "<<this->_limit<<endl;
    cout<<"-> System: "<<this->_system<<endl;
    cout<<"-> Write Path:"<<this->_writePath<<endl;
    cout<<"********************************************\n"<<endl;

    //Stampa Struttura dati letta
    map<MapId, Record>::iterator it = this->_data->begin();
    int counter = 0;

    while (it != this->_data->end()) {
        counter++;

        struct MapId  key   = it->first;
        struct Record value = it->second;

        cout<<"ID: ("<<key.frameNumber<<","<<key.idfFrame<<") - Value = [ "<<value.x_1<<", "<<value.y_1<<", "<<value.x_2<<", "<<value.y_2<<" ] \n";
        
        it++;
    }

    cout<<"\nTotal Number of Lines: "<<counter<<" \n"<<endl;

    map<int, int>::iterator ls = this->_avaibleTargets.begin();
    counter = 0;

    while (ls != this->_avaibleTargets.end()) {
        counter++;

        int  key   = ls->first;
        int value  = ls->second;

        cout<<"ID TARGET: "<<key<<" PRIMO FRAME TARGET: "<<value<<" \n"<<endl;
        
        ls++;
    }

    cout<<"\nTotal Number of Target: "<<counter<<" \n"<<endl;

}

//Metodo Lettura Dati
void DTM::DataManager::readData () {

    //Puntatore al File e Apertura
    fstream fp;
    fp.open( this->_dataPath , ios::in ); 

    //Lettura File
    if (fp.is_open()) {   

        //Stringa rappresentate la linea letta attualmente
        string tp;

        //Lettura dell'i-esima linea del file di testo
        while(getline(fp, tp)) { 

            //Strutture dati necessarie alla tokenizzazione
            char* line      = (char*) tp.data();
            const char s[2] = ",";
            char*      token;

            //Tokenizzazione string a con delimiter ","
            token = strtok(line, s);

            //Variabili di output per analisi prima riga
            int tk_counter        = 0;

            int    idx              = DTM::EMPTY_INTEGER;
            struct MapId              key;
            struct Record             rec;
            int    identifier       = DTM::EMPTY_INTEGER;
            double confidence       = DTM::EMPTY_DOUBLE;

            //Scansione Token Linea
            while( tk_counter < 7 ) {
                
                //Numero Frame
                if (tk_counter == 0) {
                    key.frameNumber = atoi(token);
                }

                //ID Frame
                if (tk_counter == 1) {
                    key.idfFrame = atoi(token);
                }

                //x_1 - Gli angoli in alto a sinistra sono considerati come (1,1)
                if (tk_counter == 2) {
                    rec.x_1  = (atoi(token)) - 1;
                }

                //y_1 - Gli angoli in alto a sinistra sono considerati come (1,1)
                if (tk_counter == 3) {
                    rec.y_1  = (atoi(token)) - 1;
                }

                //x_2
                if (tk_counter == 4) {
                    if (this->_system == "Inc") {
                        rec.x_2  = rec.x_1 + atoi(token);
                    }else{
                        rec.x_2  = atoi(token);
                    }
                }

                //y_2
                if (tk_counter == 5) {
                    if (this->_system == "Inc") {
                        rec.y_2  = rec.y_1 + atoi(token);
                    }else{
                        rec.y_2  = atoi(token);
                    }
                }

                // Prelievo Confidenza
                if (tk_counter == 6) {
                    confidence  = atof(token);
                }

                // Incremento contatore token della linea letta
                tk_counter++;
                token = strtok(NULL, s);
            }


            //Eventuale inserimento dei dati nella struttura dati di base
            if (this->_threshold == DTM::EMPTY_DOUBLE || confidence >= this->_threshold) {

                pair<MapId,Record> p (key, rec);
                this->_data->insert( p );
            }
        }

        //Chiusura puntatore file
        fp.close(); 
    }

    if (this->_data->empty() == false && this->_limit != DTM::EMPTY_INTEGER ) {

        map<MapId, Record>::iterator it = this->_data->begin();
        map<MapId,Record>*   new_data   = new map<MapId,Record>();

        int counter = 0;
        int old_key = DTM::EMPTY_INTEGER;

        while (it != this->_data->end()) {
            
            struct MapId  key   = it->first;
            struct Record value = it->second;

            //Eventuale Aggiornamento della chiave e reset del contatore
            if (old_key == DTM::EMPTY_INTEGER || old_key != key.frameNumber) {
                old_key = key.frameNumber;
                counter = 0;
            }

            if (counter < this->_limit) {
                pair<MapId,Record> p (key, value);
                new_data->insert( p );
            }
            
            counter++;
            it++;
        }

        this->_data = new_data;
    }

}

void DTM::DataManager::writeLine (int pFrameNumber, int pIdx, vector<float> pBbox, string pEmplSystem) {
    struct MapId key;
    struct Record value;

    key.frameNumber = pFrameNumber;
    key.idfFrame   = pIdx;

    int i = 0;

    for (float item:pBbox) {

        if (i==0) {
            value.x_1  = int(item);
        }

        if (i==1) {
            value.y_1  = int(item);
        }

        if (i==2) {
            if (pEmplSystem != "WH") {
                value.x_2  = int(item) - value.x_1;
            }else{
               value.x_2  = int(item); 
            }
        }

        if (i==3) {
            if (pEmplSystem != "WH") {
                value.y_2  = int(item) - value.y_1;
            }else{
                value.y_2  = int(item); 
            }
        }
        
        i++;
    }

    string base = "";
    base+= to_string(key.frameNumber) + "," + to_string(key.idfFrame) + "," + to_string(value.x_1) + "," + to_string(value.y_1) + "," + to_string(value.x_2) + "," + to_string(value.y_2) + "," + "1" + "," + "-1" + "," + "-1" + "," + "-1" + "\n";


    ofstream ofs;
    ofs.open (this->_writePath, std::ofstream::out | std::ofstream::app);

    ofs << base;

    ofs.close();

}

void DTM::DataManager::getAllTargets () {

    if (this->_data->empty() == false) {
        map<MapId, Record>::iterator it = this->_data->begin();
        set <int, greater <int> > s1; 

        
        while (it != this->_data->end()) {

            struct MapId  key   = it->first;

            int prec_len = s1.size();
            s1.insert(key.idfFrame);
            int current_len = s1.size();

            if (current_len > prec_len) {
                pair<int,int> record (key.idfFrame, key.frameNumber );
                this->_avaibleTargets.insert( record );
            }
            
            it++;
        }
        
    }
}


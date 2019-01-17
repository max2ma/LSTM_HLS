#include <iostream>
#include <ctime>
using namespace std;
#include <cmath>
#include "params.h"

extern 
void position(const DataType sensors[DataSize][Capa_In_Size+Infrared_In_Size], DataType p[DataSize][Output_Size]);

int main(){

	const int REP = 1000;
	const	DataType sensors[DataSize][20]={
#include "sensor_input.txt"
	};
	static DataType pos[REP][DataSize][2];
	const DataType ref[DataSize][2]={
#include "outputs.txt"
	};
	clock_t start = clock();
	for(int i=0;i<REP;i++)
	position(sensors,pos[i]);
	clock_t t = clock() - start;
	cout << "Execution completed"<<endl;
	cout << "Execution time "<< (double)t /CLOCKS_PER_SEC <<" s"<<endl;
	int err = 0;
	for(int r=0;r<REP;r++)
	for(int i=0;i<DataSize;i++)
		for(int j=0;j<2;j++){
			if(fabs(pos[r][i][j]/ref[i][j] -1) > 0.01){
				cout <<"pos["<<i<<"]["<<j<<"]="<<pos[i][j]<<'\t'
					<<"ref["<<i<<"]["<<j<<"]="<<ref[i][j]
					<<endl;

				err  ++;
			}
		}
	cout << "There are in total "<<err<<" errors."<<endl;
	return 0;
}

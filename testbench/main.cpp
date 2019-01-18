#include <iostream>
using namespace std;
#include <cmath>
#include "params.h"

extern 
void position(const DataType *s, DataType  *p, const int samples);

int main(){

	const	DataType sensors[DataSize][20]={
#include "sensor_input.txt"
	};
	DataType pos[DataSize][2];
	const DataType ref[DataSize][2]={
#include "outputs.txt"
	};
	int samples = 50;
	position(&sensors[0][0],&pos[0][0], samples);
	int err = 0;
	for(int i=0;i<samples;i++)
		for(int j=0;j<2;j++){
			if(fabs(pos[i][j]/ref[i][j] -1) > 0.01){
				cout <<"pos["<<i<<"]["<<j<<"]="<<pos[i][j]<<'\t'
					<<"ref["<<i<<"]["<<j<<"]="<<ref[i][j]
					<<endl;

				err  ++;
			}
		}
	cout << "There are in total "<<err<<" errors."<<endl;
	return 0;
}

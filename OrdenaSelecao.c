#include <stdio.h>
#include <time.h>
#include <stdlib.h>


void selecao(int *num, int tam) { 
  int i, j, min, aux;
  for (i = 0; i < (tam-1); i++) 
  {
     min = i;
     for (j = (i+1); j < tam; j++) {
       if(num[j] < num[min]) 
         min = j;
     }
     if (i != min) {
       aux = num[i];
       num[i] = num[min];
       num[min] = aux;
     }
  }
}

void insercao(int *num, int tam){
    int i, j, chave;

    for(i=1; i< tam; i++){
        chave = num[i];
        j=i-1;
        while(j>=0 && num[j]>=chave){
            num[j+1]= num[j];
            j = j-1;
        }
        num[j+1]= chave;
    }

}

void intercala(int *num, int p, int q, int r){
    int *num2;
    int i=p;
    int j=q;
    int k=0;

    num2 = malloc((r-p)*sizeof(int));

    while(i<q && j<r){
      if(num[i] <= num[j]){
        num2[k++] = num[i++];
      }else{
        num2[k++] = num[j++];
      }
    }

    while(i<q){
      num2[k++]=num[i++];
    }


    while(i<r){
      num2[k++]=num[j++];
    }

    for(i=p; i<r; i++){
      num[i]=num2[i-p];
    }
    free(num2);

}

void mergeSort(int *num, int p, int r){
    int q;
    if(p<r){
      q = (p+r)/2;
      mergeSort(num, p, q);
      mergeSort(num, q+1, r);
      intercala(num, p, q, r);
    }

}


void arruma(int *num, int a, int b){

    int aux1, heapMax, aux2;

    aux1 =0;
    while((a*2 <= b) && (!aux1)){
        if(a*2 == b){
          heapMax = a*2;
        }else if(num[a*2]> num[a*2 +1]){
          heapMax = a*2;
        }else{
          heapMax = a*2 + 1;
        }
    }

    if(num[a]<num[heapMax]){
      aux2 = num[a];
      num[a]= num[heapMax];
      num[heapMax]= aux2;
      a = heapMax;
    }else{
      aux1=1;
    }

}

void heapSort(int *num, int n){
    int i, aux;
    
    for(i=(n/2); i>=0;i--){
      arruma(num, i, n-1);
    }

    for(i=n-1; i>=1; i--){
      aux = num[0];
      num[0]=num[i];
      num[i]= aux;
      arruma(num, 0, i-1);
    }

}


int part(int *num, int p, int r){
    int i = p-1;
    int x = num[r];
    int j;
    int aux;

    for(j=p; j<= r-1; j++){
      if(num[j] <= x){
        i=i+1;
        aux = num[i];
        num[i] = num[j];
        num[j] = aux;
      }
    }

    aux = num[i+1];
    num[i+1] = num[r];
    num[r] = aux;
    return i+1;
}


void quickSort(int *num, int p, int r){
    int q;
    if(p<r){
      q= part(num, p, r);
      quickSort(num, p, q-1);
      quickSort(num, q+1, r);
    } 
}



int main(){

    int num[10] = {6, 4, 2, 7, 5, 1, 0, 8, 9, 3};
    int i;
    clock_t Ti, Tf;

    printf("Vetor original: ");
    for(i=0; i<10; i++){
        printf("%d ", num[i]);
    }

    Ti = clock();

    //selecao(num, 10);
    //insercao(num,10);
    mergeSort(num, 0, 10);
    //heapSort(num, 10);
    //quickSort(num,0, 10);


    Tf = clock();

    printf("\n\nvetor novo:     ");
    for(i=0; i<10; i++){
        printf("%d ", num[i]);
    }
    printf("\n\ntempooo= %f\n\n", ((float)Tf - (float)Ti));

    return 0;
}
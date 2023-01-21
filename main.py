import numpy as np
import random as rnd
import operator
import pandas as pd       
import matplotlib.pyplot as plt  
from multiprocessing import Process
from PIL import Image, ImageDraw 



class Miasto:       #Klasa definiujaca pojedyncze miasto z metoda obliczenia odleglosci miedzy tym miastem a dowolnym innym
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def odleglosc(self, city):
        xd = abs(self.x - city.x)
        yd = abs(self.y - city.y)
        return np.sqrt((xd**2) + (yd**2))

    def __print__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"



def Naiwny(listaMiast): #algorytm porownawczy (od miasta idziemy do najblizszego az nie odwiedzimy wszystkich miast)
    droga = 0
    miastaWykorzystane = []
    miastaWykorzystane.append(listaMiast[0])
    
    
    indx = 0
    logic = True
    while(logic):
        l = []
        lista_indexow = []
        for i, miasto in enumerate(listaMiast): 
            if miasto not in miastaWykorzystane:
                l.append(miasto.odleglosc(listaMiast[indx]))
                lista_indexow.append(i)
                
        a = l.index(min(l))
        indx = lista_indexow[a]
        
        miastaWykorzystane.append(listaMiast[indx]) 
        droga += min(l) 
        if(len(miastaWykorzystane) == len(listaMiast)):
            logic = False 
            
    droga += listaMiast[indx].odleglosc(listaMiast[0])
    return droga, miastaWykorzystane              
             

    



class Fitness:    #Klasa definiujaca funkcje przystosowania fitness dziala na zasadzie obliczania dlugosci trasy, nastepnie zdefiniowana jako jej odwrotnosc
    def __init__(self, droga):
        self.droga = droga
        self.odleglosc = 0
        self.fitness = 0.0

    def ZwracaOdleglosc(self):
        if self.odleglosc == 0:
            tmpOdleglosc = 0
            
            for i in range(0, len(self.droga)):
                odMiasta = self.droga[i]
                doMiasta = None
                if i + 1 < len(self.droga):
                    doMiasta = self.droga[i+1]
                else:
                    doMiasta = self.droga[0]

                tmpOdleglosc += odMiasta.odleglosc(doMiasta)

            self.odleglosc = tmpOdleglosc
        return self.odleglosc


    def ZwracaFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.ZwracaOdleglosc())   #Funkcja przystosowania jako odwrotnosc drogi
        return self.fitness                  




def generujeDroge(listaMiast):  #Funkcja generujaca pojedyncza trase (kolejnosc miast)
    droga = rnd.sample(listaMiast, len(listaMiast))
    return droga  


def generujePopulacjePoczatkowa(popSize, listaMiast): #Funkcja do generacji populacji poczatkowej
    population = []
    for i in range(0, popSize):
        population.append(generujeDroge(listaMiast))
    return population              


def rankingDrog(population): #Funkcja zwracajaca ranking drog w populacji wzgledem funkcji przystosowania w postaci uporzadkowanej listy fintess wraz z ID drogi
    fitnessResult = {}
    for i in range(0, len(population)):
        fitnessResult[i] = Fitness(population[i]).ZwracaFitness()
    return sorted(fitnessResult.items(), key = operator.itemgetter(1), reverse = True)   


def selection(populationRanking, elityzm): #Funkcja zwracajaca na podstawie outputu z rankingDrog() na podstawie wyboru ruletki liste ID drog wybranych do selekcji
    wynikSelekcji = []
    df = pd.DataFrame(np.array(populationRanking), columns=["Index","Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0,elityzm):
        wynikSelekcji.append(populationRanking[i][0])

    for i in range(0, len(populationRanking) - elityzm):
        wybor = 100*rnd.random()
        for i in range(0, len(populationRanking)):
            if wybor <= df.iat[i,3]:
                wynikSelekcji.append(populationRanking[i][0])
                break

    return wynikSelekcji   


def pulaRodzicielska(population, wynikSelekcji): #Funkcja zwraca liste rodzicow z populacji wybranych na podstawie selekcji 
    rodzice = []
    for i in range(0, len(wynikSelekcji)):
        index = wynikSelekcji[i]
        rodzice.append(population[index])
    return rodzice 


def crossover(parent1, parent2, mechanizm): #Funkcja na podstawie 2 rodzicow przeprowadza crossower poprzez 1 z 3 mechanizmow do wyboru
    child = []
    childP1 = []
    childP2 = []

    match mechanizm:
        case "Ordered Crossover":
            genA = int(rnd.random() * len(parent1))
            genB = int(rnd.random() * len(parent1))

            startGen = min(genA, genB)
            endGen = max(genA, genB)

            for i in range(startGen, endGen):
                childP1.append(parent1[i])

            childP2 = [item for item in parent2 if item not in childP1]
            child = childP1 + childP2
        
        case "Cycle Crossover":
            for i in range(len(parent1)):
                child.append(0)

            indx = 0
            logic = True
            while(logic):
                child[indx] = parent1[indx]
                indx = parent1.index(parent2[indx])
                if(child.count(parent1[indx]) != 0):
                    logic = False 

            for i in range(len(parent2)):
                if(child[i] == 0):
                    child[i] = parent2[i]

        case "Partially Mapped":
            genA = int(rnd.random() * len(parent1))
            genB = int(rnd.random() * len(parent1))

            startGen = min(genA, genB)
            endGen = max(genA, genB)
            
            for i in range(len(parent1)):
                child.append(0)

            for i in range(startGen, endGen):    
                child[i] = parent2[i]
            for i in range(startGen):
                if(child.count(parent1[i]) == 0):
                    child[i] = parent1[i]
            for i in range(endGen, len(parent1)):
                if(child.count(parent1[i]) == 0):
                    child[i] = parent1[i]
            

            for i in range(child.count(0)):
                indx = child.index(0)
                i_temp = parent2.index(parent1[indx])
                logic = True
                while(logic):
                    if(child.count(parent1[i_temp]) == 0):
                        child[indx] = parent1[i_temp]
                        logic = False
                    i_temp = parent2.index(parent1[i_temp])         
                         
                         
    return child


def nextPopulation(rodzice, elityzm, mechanizm): #Funkcja tworzaca nowa generacje na podstawie crossoveru i puli rodzicielskiej
    children = []
    length = len(rodzice) - elityzm
    pula = rnd.sample(rodzice, len(rodzice))

    for i in range(0,elityzm):
        children.append(rodzice[i])
    
    for i in range(0, length):
        child = crossover(pula[i], pula[len(rodzice) - i - 1], mechanizm)
        children.append(child)
    return children 


def mutacja(osobnik, probMutacji): #Funkcja odpowiedzialna za mutacje 1 drogi z prawdopodobienstwem probMutacji zamienia miejscami losowe 2 miasta 
    for swapped in range(len(osobnik)):
        if(rnd.random() < probMutacji):
            swapWith = int(rnd.random() * len(osobnik))

            miasto1 = osobnik[swapped]
            miasto2 = osobnik[swapWith]

            osobnik[swapped] = miasto2
            osobnik[swapWith] = miasto1

    return osobnik            


def zmutowanaPopulacja(population, probMutacji): #Funkcja na podstawie funkcji mutacja zwraca zmutowana populacje
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutacja(population[ind], probMutacji)
        mutatedPop.append(mutatedInd)

    return mutatedPop  


def nastepnaGneneracja(obecnaGeneracja, elityzm, probMutacji, mechanizm): #Funkcja korzystajaca z poprzednich w celu odtworzenia procesu selekcji, krzyzowania i mutacji i tworzaca nowa generacje
    rankingPopulacji = rankingDrog(obecnaGeneracja)
    wynikiSelekcji = selection(rankingPopulacji, elityzm)
    pula = pulaRodzicielska(obecnaGeneracja, wynikiSelekcji)  
    children = nextPopulation(pula, elityzm, mechanizm)
    nextGen = zmutowanaPopulacja(children, probMutacji) 
    return nextGen   


def algorytm(population, popSize, elityzm, probMutacji, liczbaPokolen, mechanizm, y, poczatkoweDrogi, koncoweDrogi): # Funkcja przeprowadzajaca algorytm z okreslona liczba pokolen zwracajaca najlepsza droge
    pop = generujePopulacjePoczatkowa(popSize, population)
    progress = []
    progress.append(1 / rankingDrog(pop)[0][1])
    poczatkoweDrogi.append(pop[rankingDrog(pop)[0][0]])
    #print("Poczatkowa dlugosc drogi: " + str(1 / rankingDrog(pop)[0][1]))

    for i in range(0,liczbaPokolen):
        pop = nastepnaGneneracja(pop, elityzm, probMutacji, mechanizm)
        progress.append(1 / rankingDrog(pop)[0][1])

    #print("Ostateczna dlugosc drogi: " + str(1 / rankingDrog(pop)[0][1]))
    #indxNajlepszejDrogi = rankingDrog(pop)[0][0]
    #najlepszaDroga = pop[indxNajlepszejDrogi]
    koncoweDrogi.append(pop[rankingDrog(pop)[0][0]])

    #plt.plot(progress)
    #plt.ylabel("Odleglosc")
    #plt.xlabel("Pokolenie")
    #plt.show()
    #return najlepszaDroga
    y.append(progress)

def wykres(lista_y):
    plt.plot(lista_y)
    plt.ylabel("Odleglosc")
    plt.xlabel("Pokolenie")
    plt.show()


def rysuj(listaMiast, Droga):
    SCALE = 4
    im = Image.new("RGB", (200 * SCALE, 200 * SCALE), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    DIAMETER = 10 * SCALE
    
    for city in listaMiast:
        draw.ellipse((city.x*SCALE - DIAMETER/2, city.y*SCALE - DIAMETER/2, city.x*SCALE + DIAMETER/2, city.y*SCALE + DIAMETER/2), fill = (255, 0, 0), outline = (0, 0, 0))
    for i in range(len(listaMiast) - 1):
        draw.line((Droga[i].x*SCALE, Droga[i].y*SCALE, Droga[i+1].x*SCALE, Droga[i+1].y*SCALE), fill = (0, 0, 255), width = 5)
    draw.line((Droga[0].x*SCALE, Droga[0].y*SCALE, Droga[len(listaMiast)-1].x*SCALE, Droga[len(listaMiast)-1].y*SCALE), fill = (0, 0, 255), width = 5)

    #im = im.thumbnail(800, 800)
    im.show()    
                                  


def main():
    lICZBA_MIAST = 30
    LICZEBNOSC_POKOLENIA = 100
    ELITYZM = 20
    LICZBA_POKOLEN = 500    
    PRAWDOPODOBIENSTWO_MUTACJI = 0.01
    MECHANIZM = "Ordered Crossover"
    ILOSC_PRZEPROWADZONYCH_ALGORYTMOW = 1
    listaMiast = []

    for i in range(0, lICZBA_MIAST):
        listaMiast.append(Miasto(x=int(rnd.random() * 200), y=int(rnd.random() * 200)))

    y = []
    sr_droga = []
    poczatkowe_drogi = []
    koncowe_drogi = []
    for i in range(LICZBA_POKOLEN):
        sr_droga.append(0) 
       

    processes = []
    for i in range(ILOSC_PRZEPROWADZONYCH_ALGORYTMOW):
        processes.append(Process(target = algorytm(population=listaMiast, popSize = LICZEBNOSC_POKOLENIA, elityzm = ELITYZM, probMutacji = PRAWDOPODOBIENSTWO_MUTACJI, liczbaPokolen = LICZBA_POKOLEN, mechanizm = MECHANIZM, y = y, poczatkoweDrogi = poczatkowe_drogi, koncoweDrogi = koncowe_drogi)))
    
    for p in processes:
        p.start()
        

    for p in processes:
        p.join()

    for i in range(ILOSC_PRZEPROWADZONYCH_ALGORYTMOW):
        for j in range(LICZBA_POKOLEN):
            sr_droga[j] += y[i][j]

    for i in range(LICZBA_POKOLEN):
        sr_droga[i] = sr_droga[i]/ILOSC_PRZEPROWADZONYCH_ALGORYTMOW        


    print("Srednia poczatkowa dlugosc drogi: " + str(sr_droga[0]))
    print("Srednia koncowa dlugosc drogi" + str(sr_droga[LICZBA_POKOLEN - 1]))
    print("Dlugosc drogi dla algorytmu Naiwnego: " + str(Naiwny(listaMiast)[0]))
    wykres(sr_droga)

    PoczatkowaDroga = poczatkowe_drogi[0]
    KoncowaDroga = koncowe_drogi[0]
    
    rysuj(listaMiast, PoczatkowaDroga)
    rysuj(listaMiast, KoncowaDroga)      
    rysuj(listaMiast, Naiwny(listaMiast)[1])


if __name__ == "__main__":
    main()              
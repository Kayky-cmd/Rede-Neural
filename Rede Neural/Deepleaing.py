

import numpy as np

class Rede_Neural():
    
    def __init__(self):
        self.n_entradas = 2
        self.n_camadas = 3
        self.n_saidas = 1
        self.tr = 0.7
        
        self.w_1 = np.random.rand(self.n_camadas, self.n_entradas) - 0.5
        self.w_2 = np.random.rand(self.n_saidas, self.n_camadas) - 0.5
        
        self.bias_h = np.random.rand(self.n_camadas, 1) - 0.5
        self.bias_o = np.random.rand(self.n_saidas, 1) - 0.5
    
        self.sig = lambda x: 1/(1+np.exp(-x))
        self.dsig = lambda y: y*(1.0-y)
    

    def backpropagation(self, x_tr, y_alvo):
        x = np.array(x_tr, ndmin=2).T
        
        saida_oculta = self.sig(np.add((np.dot(self.w_1, x)), self.bias_h))
        saida_final = self.sig(np.add((np.dot(self.w_2, saida_oculta)), self.bias_o))
        
        erro = y_alvo - saida_final
        if (i % 5000) == 0:
            print('Erro:', erro)
        
        erro_oculto = np.dot(self.w_2.T, erro)
        
        self.w_2 += self.tr*(np.dot((erro*(self.dsig(saida_final))), np.transpose(saida_oculta)))
        #Ajustando o bias pelo delta, com o gradiente da curva sigmoide(apenas o gradiente).
        self.bias_o += self.tr*(erro*(self.dsig(saida_final)))
        
        self.w_1 += self.tr*(np.dot((erro_oculto*(self.dsig(saida_oculta))), np.transpose(x)))
        self.bias_h += self.tr*(erro_oculto*(self.dsig(saida_oculta)))
        
    def consultar(self, x_con):
        x = np.array(x_con, ndmin=2).T
        
        saida_oculta = self.sig(np.add((np.dot(self.w_1, x)), self.bias_h))
        saida_final = self.sig(np.add((np.dot(self.w_2, saida_oculta)), self.bias_o))
        
        return print(saida_final)

x_treinamento = np.array([[1,1],[1,0],[0,1],[0,0]])

y_alvos = np.array([[0],[1],[1],[0]])

for i in range(15000):
    for x_tr,y_alvo in zip(x_treinamento, y_alvos):
        rn.backpropagation(x_tr, y_alvo)
Erro: [[-0.51803794]]
Erro: [[0.51420212]]
Erro: [[0.49030994]]
Erro: [[-0.56600458]]
Erro: [[-0.02901454]]
Erro: [[0.01826292]]
Erro: [[0.01828853]]
Erro: [[-0.00192317]]
Erro: [[-0.01973826]]
Erro: [[0.01244559]]
Erro: [[0.01245454]]
Erro: [[-0.00113071]]

from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    for i in range(1,12):

        docs = [
            [f"/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_{i}.png"]
        ]

        Ns = [26]
        
        learning_rates = [0.01]

        central_tendencies = ["mean"]

        backward_adaptive_coding_experiment(
            docs,Ns,learning_rates,central_tendencies,
            manual_th=0.4,full_page=False,page_shape = (1024,791)
        )

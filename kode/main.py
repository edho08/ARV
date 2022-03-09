if __name__ == "__main__":
    from Trainer import DARVTrainer
    from DARV import DARV
    
    model = DARV(n=300)
    trainer = DARVTrainer(model)
    trainer.train('data/english.50MB', model, worker=6, max_data=-1, num_data_per_worker=200000)

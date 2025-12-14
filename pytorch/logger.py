import wandb

class Logging():
    def __init__(self,project,name,model,lr,batch,ep,optim):
        wandb.init(
            project=project,
            name=name,
            config={
                "model": model,
                "learning_rate": lr,
                "batch_size": batch,
                "epochs": ep,
                "optimizer": optim
            }
        )
        
    def log(self,tag,epoch,loss,acc,epoch_time,total_time,memory):
        wandb.log({
            f'epoch': epoch + 1,
            f'{tag}/loss' : loss,
            f'{tag}/acc' : acc,
            f'{tag}/epoch_time' : epoch_time,
            f'{tag}/total_time' : total_time,            
            f'{tag}/memory' : memory,
        })
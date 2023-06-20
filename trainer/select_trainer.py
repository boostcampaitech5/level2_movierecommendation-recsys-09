from trainer import Trainer, AutoRecTrainer, Trainer_ML, MVAE_Trainer, BERT4RecTrainer

def select_trainer (model, criterion, metrics, optimizer, config, device, data_loader, valid_data_loader, lr_scheduler):
    if config['name'] == "AutoRec_eval":
        trainer = AutoRecTrainer(
            model, data_loader, valid_data_loader, None, None, config)
     
    elif config['name'] == "Catboost":
        trainer = Trainer_ML(model, config=config,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader)
        
    elif config['name'] == "MVAE":
        trainer = MVAE_Trainer(model, criterion, config=config,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader, optimizer = optimizer)
    
    elif config['name'] == 'BERT4Rec':
        trainer = BERT4RecTrainer(model, config, data_loader, criterion, optimizer, lr_scheduler, device)
        
    else:
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)
        
    return trainer
import torch

def test(model, device, test_criterion,
          test_dataloader, load=False, load_path='./checkpoint.pt'):
    
    model.to(device)
    
    intent_error=0
    tag_error=0
    
    if load:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
                
    #validate
    model.eval()
    
    test_iter = 0

    
    for test_x, (test_y_label, test_y_tag) in test_dataloader:

        with torch.no_grad():

            test_x = test_x.to(device)
            test_y_label = test_y_label.to(device)
            test_y_tag = test_y_tag.to(device)
            batch_size = len(test_y_label)

            intent, tag = model(test_x)

            intent_loss = test_criterion(intent, test_y_label)

            tag_loss = test_criterion(tag.reshape((-1,tag.shape[2])), test_y_tag.reshape((-1,)))
            
            

            intent_error += intent_loss.item() * batch_size
            tag_error += tag_loss.item() * batch_size

            test_iter += batch_size

    intent_error /= test_iter
    tag_error /= test_iter
    t_loss = intent_error + tag_error
      
    return epoch, t_loss, intent_error, tag_error

        
 
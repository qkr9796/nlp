import torch

def train(model, optimizer, device, num_epoches, tag_criterion, intent_criterion, test_criterion,
          train_dataloader, valid_dataloader, save_path='./checkpoint.pt', load=False, load_path='./checkpoint.pt'):
    
    model.to(device)
    
    epoch=0
    intent_error=[]
    tag_error=[]
    val_error=[]
    best_error=float('inf')
    
    if load:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        intent_error = checkpoint['intent_error']
        tag_error = checkpoint['tag_error']
        best_error = checkpoint['best_error']
        val_error = checkpoint['val_error']
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)


    while epoch < num_epoches:

        print('Epoch : %d' % (epoch+1))
        intent_error.append(0)
        tag_error.append(0)
        val_error.append(0)
        num_iter = 0

        #train
        model.train()

        for train_x, (train_y_label, train_y_tag) in train_dataloader:

            train_x = train_x.to(device)
            train_y_label = train_y_label.to(device)
            train_y_tag = train_y_tag.to(device)

            intent, tag = model(train_x)

            intent_loss = intent_criterion(intent, train_y_label)

            tag_loss = tag_criterion(tag.reshape((-1,tag.shape[2])), train_y_tag.reshape((-1,)))

            t_loss = intent_loss + tag_loss

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            num_iter+=1

            intent_error[-1] += intent_loss.item()
            tag_error[-1] += tag_loss.item()


        intent_error[-1] /= num_iter
        tag_error[-1] /= num_iter
        print('intent error: %f, tag error: %f' % (intent_error[-1], tag_error[-1]))   
                
        #validate
        model.eval()
        
        val_intent_error = 0
        val_tag_error = 0

        val_iter = 0
        for test_x, (test_y_label, test_y_tag) in valid_dataloader:

            with torch.no_grad():

                test_x = test_x.to(device)
                test_y_label = test_y_label.to(device)
                test_y_tag = test_y_tag.to(device)

                intent, tag = model(test_x)

                intent_loss = test_criterion(intent, test_y_label)

                tag_loss = test_criterion(tag.reshape((-1,tag.shape[2])), test_y_tag.reshape((-1,)))

                val_intent_error += intent_loss
                val_tag_error += tag_loss

                val_iter += 1

        val_intent_error /= val_iter
        val_tag_error /= val_iter
        t_loss = val_intent_error + val_tag_error
        val_error[-1] = (t_loss, val_intent_error, val_tag_error)
        
        
        epoch+=1             
        
        if t_loss < best_error:
            best_error = t_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'intent_error': intent_error,
                'tag_error': tag_error,
                'best_error': best_error,
                'val_error': val_error
            }, save_path +'best')
            print('saving new best model')

        print('validation error: %f' % t_loss)

        print('saving model,optimizer parameters')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'intent_error': intent_error,
                'tag_error': tag_error,
                'best_error': best_error,
                'val_error': val_error
            }, save_path)        

        if epoch % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'intent_error': intent_error,
                'tag_error': tag_error,
                'best_error': best_error,
                'val_error': val_error
            }, save_path + 'epoch' + str(epoch))
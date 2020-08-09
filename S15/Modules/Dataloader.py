def get_data_loader(train_set,test_set,seed=1,batch_size=8,num_workers=4,pin_memory=True):
  SEED = 1
  cuda = torch.cuda.is_available()
  torch.manual_seed(SEED)
  if cuda:
    torch.cuda.manual_seed(SEED)
  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)
  train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
  test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)
  return train_loader, test_loader
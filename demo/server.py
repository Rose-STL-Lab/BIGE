import paramiko
from scp import SCPClient

import options.option_limo as option_limo

class NorthUCSDServer(paramiko.SSHClient):
    def __init__(self, *args, **kwargs):
        key_filename = kwargs['key_filename']
        del kwargs['key_filename']

        super().__init__(*args, **kwargs) 

        self.load_system_host_keys()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        key = paramiko.RSAKey.from_private_key_file(key_filename) # Convert OpenSSH private key is RSA private key: ssh-keygen -p -m PEM -f <path-to-key>  
        self.connect("north.ucsd.edu", port=12700, username="ubuntu", pkey=key,look_for_keys=False,allow_agent=False)    

        self.scp_client = SCPClient(self.get_transport())

    def state():
        pass

    def sync_experiments()
        stdin, stdout, stderr = ssh_client.exec_command('ls /data/panini/digital-coach-anwesh/output')
        online_experimenets = stdout.readlines()

        current_experiments = os.listdir(args.out_dir)
        
        


        
    def close(self):
        ssh_client.close()

if __name__ == "__main__": 
        
    args = option_limo.get_args_parser()
    torch.manual_seed(args.seed)
    # parser.add_argument('--server-key', default=None, type=str, help="Private key for the server")
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark=False


    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)

    NorthUCSDServer()
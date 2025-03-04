# WSI-Genie-personal- Writing the documentation - To run on Mac and Windows without an issue:



### Instroducation: What it is, What it does, why do we need it, how it works, System requirements (maybe an automated system checker to check if we have enough of RAM, GPUs, etc).
### Workflow of the installation and running: 
### Check if we're storing all the results in a good place, 



Clone and make config files from templates

```
git clone --recurse-submodules https://github.com/pitt-cpace/milk.git -- Clone Branch 86 : 86-pause-resume-and-cancel-processing-pipeline-from-ui (this branch works well, better than the previous 'main' branch
cd milk
cp dsa_custom_config/compose.datsetimports.yaml.example dsa_custom_config/compose.datsetimports.yaml
```

## Windows: 

Backend

```
conda env create -f backend-Windows-requirements/environment.yml 
conda activate genie
```

## Mac / Linux: 

Backend

```
conda env create -f backend-Mac-requirements/environment.yml
conda activate genie

```



## Install Docker:

## Download Docker: Select the right version Mac/Linux/Windows: 

```
https://www.docker.com/products/docker-desktop/

```

## Terminal - set-up path for Docker:

```
export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin/"
```

## Launch the APP: 

```
DSA_USER=$(id -u):$(id -g) docker compose --profile dev up --watch

```




# Frontend

```
cd frontend
npm install
npm run dev
```

Webpage will be hosted at [http://localhost:5173/](http://localhost:5173/)

## Docker instructions

WARNING: Attempting to view the files of the girder container through Docker Desktop crashes docker in way that is often unrecoverable without restarting the entire computer

### Windows/WSL

#### Configuring Backdoor imports

`dsa_custom_config/compose.datsetimports.yaml` includes space to add volumes to girder to allow for backdoor symbolic imports

#### Development

```
docker compose pull
DSA_USER=$(id -u):$(id -g) docker compose --profile dev up --build --watch
```

Webpage will be hosted at [http://localhost:8081/](http://localhost:8081/)

#### Production

```
docker compose pull
DSA_USER=$(id -u):$(id -g) docker compose up --build
```

Webpage will be hosted at [http://localhost:8081/](http://localhost:8081/)

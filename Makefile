# set user
user 			= 1000:0

# docker image and container details
Docker_name		= knoriy/audio_dataloader
container_name	= audio_dataloader

# forwards ports
ports 			= -p 8888:8888

# set volume directory
volume_dir 		= $(shell pwd):/workspace
dataset_dir		= /media/knoriy/DATA/Datasets/:/Datasets

build:
	@docker build . -t $(Docker_name)
	
noCache:
	@docker build --no-cache . -t $(Docker_name)

stop:
	@docker stop $(container_name)

run:
	@docker run -it -d --rm --gpus=all $(ports) -v $(volume_dir) -v $(dataset_dir) --name $(container_name) $(Docker_name)

bash:
	@docker run -it -d --rm --gpus=all $(ports) -v $(volume_dir) -v $(dataset_dir) --name $(container_name) $(Docker_name) bash || docker exec -it $(container_name) bash



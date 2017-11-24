#! /bin/sh
create_output_folders()
{
	mkdir outputs
	mkdir -p outputs/models
        mkdir -p outputs/results
}

create_input_folders()
{
	mkdir -p data
	mkdir -p data/train
	mkdir -p data/val
}

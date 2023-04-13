.PHONY: format
format:
	poetry run pysen run format

.PHONY: lint
lint:
	poetry run pysen run lint

.PHONY: build
build:
	docker-compose -f ./docker-compose.yml build foreigner_generation

.PHONY: up
up:
	docker-compose -f ./docker-compose.yml up -d foreigner_generation

.PHONY: exec
exec:
	docker exec -it foreigner_generation bash

.PHONY: down
down:
	docker-compose -f ./docker-compose.yml down

.DEFAULT_GOAL := show-help

include .env
export


.PHONY: deploy
deploy: stop build angular/build start

.PHONY: build
build:
	docker-compose build

.PHONY: start
start:
	docker-compose up -d apache backend

.PHONY: stop
stop:
	docker-compose down

.PHONY: logs
logs:
	docker-compose logs --follow

# Angular
.PHONY: angular/build
angular/build:
	docker-compose up angular

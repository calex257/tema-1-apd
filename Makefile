build: tema1_par.cpp
	g++ tema1_par.cpp helpers.c -o tema1_par -lm -lpthread -Wall -Wextra
debug: tema1_par.cpp
	g++ tema1_par.cpp helpers.c -o tema1_par -lm -lpthread -g -Wall -Wextra
clean:
	rm -rf tema1 tema1_par
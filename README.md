# Dimensionnement de stockage embarqué dans un tramway

## Table des Matières
- [Objectifs du projet](#objectifs)
- [Présentation du système de tramway étudié](#presentation-du-systeme-de-tramway-etudie)
- [Réalisation du projet](#realisation-du-projet)

## Objectifs du projet

Lors de ce projet d'optimisation, nous essaierons de :
- Mettre en œuvre une démarche d’optimisation pour la conception d’un système,
- Comprendre le principe de l’alimentation électrique des tramways,
- Découvrir l’optimisation multi-critères et le concept d’optimalité au  sens de Pareto,
- Et mettre en œuvre l’algorithme génétique `NSGA-2`. 

## Présentation du système de tramway étudié

Ce projet porte sur l’étude en simulation d’une ligne de tramway très simple, et plus précisément sur l’intérêt d’un système de stockage par batterie à bord afin de faire de la récupération d’énergie au freinage. On souhaite dimensionner et piloter un système de stockage de façon à réduire les chutes de tension aux bornes du train. Pour cela, il faut réaliser un compromis entre coût et performance du système de stockage.

## Réalisation du projet

### 1. Mise en place du modèle de système

- Ajout d'une simulation d'une mise en mouvement d'un train sans batterie, puis avec batterie.


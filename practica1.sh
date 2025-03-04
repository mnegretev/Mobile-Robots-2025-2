
cd ~/catkin_ws




LOG_FILE="results.log"
echo "Iniciando pruebas de planeación de rutas" > $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE


cost_radii=("0.05")
start_positions=("5 5" "2.0 10.0" "10.5 9.75" "2.0 7.0")
diagonal_options=("True" "False")

run_test() {
    local cost_radius=$1
    local start_x=$2
    local start_y=$3
    local diagonals=$4

    echo "Ejecutando prueba con cost_radius=$cost_radius, start=($start_x, $start_y), diagonals=$diagonals"
    echo "----------------------------------------" >> $LOG_FILE
    echo "Cost Radius: $cost_radius" >> $LOG_FILE
    echo "Start Position: ($start_x, $start_y)" >> $LOG_FILE
    echo "Diagonals: $diagonals" >> $LOG_FILE

    
    #rosrun path_planner cost_map.py cost_radius:=$cost_radius

    
    { time rosrun path_planner a_star.py start_x:=$start_x start_y:=$start_y diagonals:=$diagonals; } 2>> $LOG_FILE

    echo "----------------------------------------" >> $LOG_FILE
}


for cost_radius in "${cost_radii[@]}"; do
    for start_pos in "${start_positions[@]}"; do
        for diagonals in "${diagonal_options[@]}"; do
            run_test $cost_radius $start_pos $diagonals
        done
    done
done

echo "Pruebas finalizadas. Resultados guardados en $LOG_FILE"

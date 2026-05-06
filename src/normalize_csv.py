import csv

INPUT_FILE = 'trajectory.csv'   # Укажите имя вашего входного файла
OUTPUT_FILE = 'trajectory_norm.csv' # Укажите имя выходного файла

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f_out:
    
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    for row in reader:
        if not row:  # Пропуск пустых строк, если они есть
            continue
            
        # Предполагаем, что столбец с 'frame_...' последний
        last_col = row[-1]
        
        if last_col.startswith('frame_'):
            # Убираем префикс, преобразуем в int (убирает ведущие нули), обратно в str
            row[-1] = str(int(last_col.replace('frame_', '')))
            
        writer.writerow(row)

print(f'✅ Преобразование завершено. Результат сохранён в {OUTPUT_FILE}')

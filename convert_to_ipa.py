import eng_to_ipa
import sys

def convert_to_ipa(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                try:
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue

                    # Split the line into filename and transcript
                    if '|' not in line:
                        print(f"Warning: Line {line_num} does not contain '|', skipping: {line}")
                        continue
                    
                    filename, transcript = line.split('|', 1)
                    
                    # Convert transcript to IPA
                    ipa_text = eng_to_ipa.convert(transcript)
                    
                    # Write the converted line with |0 appended
                    output_line = f"{filename}|{ipa_text}|0\n"
                    f_out.write(output_line)

                    # Print progress every 100 lines
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} lines...")

                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
                
        print(f"Conversion completed. Output written to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # First install eng_to_ipa if not already installed
    try:
        import eng_to_ipa
    except ImportError:
        print("Installing required package eng_to_ipa...")
        import pip
        pip.main(['install', 'eng_to_ipa'])
        import eng_to_ipa
    
    convert_to_ipa(input_file, output_file)
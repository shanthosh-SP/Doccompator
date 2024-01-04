import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { trigger, style, animate, transition } from '@angular/animations';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
  animations: [
    trigger('scrollAnimation', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('300ms', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class HomeComponent {
  showCompage = false;
  response:any;
  rawResponse:any;
  changeList:any = [];
  changeListFinalData:any = [];
  selectedIndex:any;
  showLoader = false;
  fileIcon:any = '';
  showFileMatchScore = false;

  file1: any;
  file2: any;
  file1Name:any;
  file2Name:any;
  constructor(
    private http: HttpClient,
    private sanitizer: DomSanitizer
  ){
  //  this.getData();
    this.file1 = null;
    this.file2 = null;
  }

  onFile1Selected(event: any) {
    this.file1 = event.target.files[0];

    const fileName = this.file1.name.toLowerCase();
    this.file1Name = this.file1.name;
    const isPpt = fileName.endsWith('.ppt') || fileName.endsWith('.pptx');
    const isPdf = fileName.endsWith('.pdf');
    const isExcel = fileName.endsWith('.xls') || fileName.endsWith('.xlsx');

    if (isPpt) {
      this.fileIcon = '/assets/ppt.png';
    } else if (isPdf) {
      this.fileIcon = '/assets/pdf.png';
    } else if (isExcel) {
      this.fileIcon = '/assets/xls.png';
    } else {
      console.log('Selected file is not a supported type.');
    }
  
  }

  onFile2Selected(event: any) {
    this.file2 = event.target.files[0];
    this.file2Name = this.file2.name;
  }

  getData(){
    this.showLoader = true;
    const formData = new FormData();
    formData.append('file1', this.file1);
    formData.append('html_file', this.file2);

    // const headers = new HttpHeaders({
    //   'Content-Type': 'multipart/form-data',
    // });

    this.http.post('http://localhost:61077/process_and_compare_files',formData).subscribe((res:any)=>{
    this.showLoader = false;
    this.showCompage = true;
      this.rawResponse = JSON.parse(JSON.stringify(res));
      this.changeList = this.splitChangeContent(res.comparison_output.content);
      this.response = res;
     });
  }

  formateResponse(val:any){
    if(val){
      return val.replace(/\n/g, '<br>')
    } else {
      return "";
    }
     
  }

  splitChangeContent(res:any){
    let splitParentData = res.split("\n\n");
    if(splitParentData.length){
      splitParentData.forEach((element:any) => {
      if(this.splitSubContent(element)){
        this.changeListFinalData.push(this.splitSubContent(element));
      }
        
      });
    }
    return this.changeListFinalData;
  }

  splitSubContent(val:any){
    let data = val.split("\n");
    let retunObj:any = {};
    data.forEach((ele:any) =>{
       let objVal = ele.split(":");
       let key = objVal[0];
       let value = objVal[1];
       if(key && value){
        if(key=='Line Content'){
          retunObj.line_content = this.removeLeadingWhitespace(value);
        } else {
          retunObj[key] = this.removeLeadingWhitespace(value);
        }
       
       }
    
    })
  return retunObj;
  }


   highlightText(data:any) {
    let change = JSON.parse(JSON.stringify(data));

    // Get the input string and the target text
   let rawContent = JSON.parse(JSON.stringify(this.rawResponse.pdf_text));
    // Check if the input string exists in the target text
    // if (!rawContent.includes(change.line_content)) {
      // Replace the target text with the highlighted version

      const replacedContent = this.replaceWordByPosition(change.line_content,parseInt(change.Position),`<span class="highlight" id="highlightElement" [@scrollAnimation]>${change.Word}</span>`,change.Word);

      // const replace_word = change.line_content.replace(new RegExp(change.Word, 'g'),
      // `<span class="highlight" id="highlightElement" [@scrollAnimation]>${change.Word}</span>`);

      const highlightedText = rawContent.replace(change.line_content,replacedContent);

      
      // const highlightedText = rawContent.replace(
      //   new RegExp(change.line_content, 'g'),
      //   `

      // );
    //  console.log(highlightedText);
      
      this.response.pdf_text = this.formateResponse(highlightedText);
    //  console.log(this.response.pdf_text);

      setTimeout(() => {
        this.scrollToElement();
      }, 20);

     

    // } else {
    //  console.log('not there');
    // }
  }


  replaceWordByPosition(paragraph:any, position:any, replacement:any, word:any) {
    // Split the paragraph into an array of words
    const words = paragraph.split(/\s+/);
    console.log(JSON.parse(JSON.stringify(words)));
    // Ensure the position is within the valid range
    if (position > words.length) {
        console.error('Invalid position');
        return paragraph;
    }

    // Replace the word at the specified position
    words[position] = words[position].replace(word,replacement);

    // Join the words back into a paragraph
    const updatedParagraph = words.join(' ');

    return updatedParagraph;
}


  removeLeadingWhitespace(inputString:any) {
    return inputString.replace(/^\s+/, '');
  }

  scrollToElement(): void {
    const element = document.querySelector(".highlight");
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  }


  goBack(){
   this.showCompage = false;
   this.showFileMatchScore = false;
   this.fileIcon = '';
   this.changeList = [];
   this.rawResponse = null;
   this.response = null;
   location.reload();
  }

  safeHtml(val:any){
    return this.sanitizer.bypassSecurityTrustHtml(val)
  }

}
